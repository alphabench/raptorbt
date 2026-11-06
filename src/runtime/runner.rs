//! The multi-strategy core: one tape, many isolated books.
//!
//! [`RuntimeCore`] owns the shared pieces of a run (the [`MarketStore`], the
//! bar-aggregator hub, the indicator bank and the subscription index) and one
//! [`PortfolioBook`] per strategy. It is plain Rust: a driver (the Python
//! bindings, or a test) walks the tape with a [`MergeCursor`] and calls
//!
//! 1. [`RuntimeCore::advance`] once per event, which moves the visibility
//!    watermark, pushes the event through the aggregators and feeds the
//!    indicator bank (`begin_event`, `emit_composite`, `emit_primary`). An
//!    event is a bar, a trade print, a quote or a depth snapshot; a print
//!    feeds the aggregators as `BarAggregator.push_trade` would, and when it
//!    completes a primary bar (see [`RuntimeBuilder::set_primary_bars`]) the
//!    bank gets that bar's primary emission;
//! 2. for each strategy in [`RuntimeCore::subscribers`] order, whatever
//!    strategy code it runs, revealing indicator emissions as it consumes
//!    them (see [`crate::runtime::indicators`]);
//! 3. [`RuntimeCore::apply`] to step that strategy's book with the event.
//!
//! # Isolation
//!
//! A strategy has its own account: no cash, margin or halt is shared, unless
//! it is put in an account group (see [`crate::runtime::groups`]), where the
//! members of the group share one. What is shared is read-only market data and the pure functions of it
//! (aggregated bars, indicator values, ATR series).
//!
//! # Per-instrument ordinals
//!
//! A book addresses instruments by *local* index: the position of the
//! instrument in the strategy's own subscription list, which is kept in
//! ascending global-id order, i.e. the order a standalone session over the
//! same symbols would register them. The event ordinal (`local_idx`) is the
//! cursor's per-instrument counter, which counts that instrument's events
//! only; with a bars-only store it equals the bar's row, with ticks it counts
//! every bar, trade, quote and book of the instrument. Neither depends on
//! which other strategies or instruments exist, so a strategy subscribed to a
//! subset sees the ordinals, and the tie-break order at equal timestamps
//! (stream = instrument id), its standalone session would.

use std::collections::HashMap;
use std::sync::Arc;

use crate::accounts::AccountMode;
use crate::core::error::RaptorError;
use crate::core::types::{BacktestConfig, Direction, InstrumentConfig, OhlcvBar};
use crate::core::Result;
use crate::data::events::DepthRef;
use crate::data::{BarSpec, BarStamp};
use crate::instruments::InstrumentSpec;
use crate::portfolio::book::PortfolioBook;
use crate::portfolio::kernel::Adoption;
use crate::portfolio::kernel::{EngineEvent, KernelBar, StepInput};
use crate::portfolio::ledger::PositionPolicy;
use crate::portfolio::session::{ScheduleData, ScheduleEntry, SessionOutcome};
use crate::quant::eval::{ClockCtx, EvalError};
use crate::runtime::aggregators::{AggKey, AggregatorHub, AggregatorHubBuilder};
use crate::runtime::atr_cache::AtrCache;
use crate::runtime::cursor::{EventKind, EventRef, MergeCursor};
use crate::runtime::exprs::ProgramId;
use crate::runtime::groups::{AccountGroup, GroupOutcome, GroupSpec};
use crate::runtime::ids::{AggId, InstrumentId, StrategyId};
use crate::runtime::indicators::{
    IndInput, IndKey, IndKind, IndicatorBank, IndicatorRegistry, LocalStreamId, Ohlc, PrimaryOrder,
    Registration,
};
use crate::runtime::memory::HeapSize;
use crate::runtime::router::{SubscriptionIndex, SubscriptionIndexBuilder};
use crate::runtime::store::MarketStore;

/// How the books treat one instrument.
#[derive(Debug, Clone)]
pub struct InstrumentParams {
    pub direction: Direction,
    pub spec: Option<InstrumentSpec>,
    pub config: Option<InstrumentConfig>,
    /// ATR period behind ATR-based stops and targets, `0` for none.
    pub atr_period: usize,
}

impl Default for InstrumentParams {
    fn default() -> Self {
        Self { direction: Direction::Long, spec: None, config: None, atr_period: 0 }
    }
}

/// A strategy's engine-side record.
///
/// Its instruments are `strat_insts[inst_start..inst_start + inst_len]`.
#[derive(Debug, Clone, Copy)]
pub struct StrategySlot {
    inst_start: u32,
    inst_len: u32,
    /// Events applied to the strategy's book.
    events: u64,
}

/// Registration phase of a [`RuntimeCore`].
pub struct RuntimeBuilder {
    store: Arc<MarketStore>,
    params: Vec<InstrumentParams>,
    config: BacktestConfig,
    mode: AccountMode,
    policy: PositionPolicy,
    index: SubscriptionIndexBuilder,
    hub: AggregatorHubBuilder,
    registry: IndicatorRegistry,
    slots: Vec<StrategySlot>,
    strat_insts: Vec<InstrumentId>,
    books: Vec<Option<PortfolioBook>>,
    groups: Vec<AccountGroup>,
    /// Members each group was declared with, by group.
    declared: Vec<usize>,
    /// Each strategy's group.
    group_of: Vec<Option<u32>>,
    /// Positions adopted so far, `(strategy, instrument, timestamp, price,
    /// size)`, so a shadow book can be seeded with the same ones.
    adopted: Vec<(StrategyId, usize, crate::core::types::Timestamp, f64, f64, Adoption)>,
    /// The zone and what the primary bars' timestamps mean, for an expression that uses
    /// `at(...)`. Unset: such an expression is refused at [`RuntimeBuilder::build`].
    clock: Option<(i64, BarStamp)>,
    /// Primary bar widths the caller knows (a live node told `bar_ns`), ahead of the stored bars.
    widths: HashMap<InstrumentId, i64>,
}

impl RuntimeBuilder {
    /// `params` holds one entry per instrument of `store`, by id.
    pub fn new(
        store: impl Into<Arc<MarketStore>>,
        params: Vec<InstrumentParams>,
        config: BacktestConfig,
        mode: AccountMode,
        policy: PositionPolicy,
    ) -> Result<Self> {
        let store = store.into();
        if params.len() != store.len() {
            return Err(RaptorError::invalid_parameter(format!(
                "{} instrument parameter sets for {} instruments",
                params.len(),
                store.len()
            )));
        }
        let n = store.len();
        for (i, p) in params.iter().enumerate() {
            let id = InstrumentId::new(i as u32);
            if p.atr_period > 0 && (store.ticks(id).is_some() || store.depth(id).is_some()) {
                return Err(RaptorError::invalid_parameter(format!(
                    "ATR-based stop/target (period {}) on {:?}, which has ticks or depth: \
                     tick events carry no bar series to compute ATR from",
                    p.atr_period,
                    store.symbols().symbol(id).unwrap_or_default()
                )));
            }
        }
        Ok(Self {
            store,
            params,
            config,
            mode,
            policy,
            index: SubscriptionIndexBuilder::new(n),
            hub: AggregatorHubBuilder::new(n),
            registry: IndicatorRegistry::new(),
            slots: Vec::new(),
            strat_insts: Vec::new(),
            books: Vec::new(),
            groups: Vec::new(),
            declared: Vec::new(),
            group_of: Vec::new(),
            adopted: Vec::new(),
            clock: None,
            widths: HashMap::new(),
        })
    }

    /// Say how the primary bars' timestamps are read: `tz_offset_ns` is the zone daily windows
    /// are cut in (an instrument's own session replaces it), `stamp` whether a timestamp opens
    /// or closes its bar. An expression cell using `at(...)` takes both from here.
    /// Tell the builder the width of `inst`'s primary bars, in nanoseconds.
    pub fn set_bar_width(&mut self, inst: InstrumentId, width_ns: i64) {
        if width_ns > 0 {
            self.widths.insert(inst, width_ns);
        }
    }

    pub fn set_clock(&mut self, tz_offset_ns: i64, stamp: BarStamp) {
        self.clock = Some((tz_offset_ns, stamp));
    }

    /// The clock the cell of `key` runs on, for an expression that uses `at(...)`.
    fn cell_clock(&self, key: &IndKey) -> Result<Option<ClockCtx>> {
        if key.kind != IndKind::Expr {
            return Ok(None);
        }
        self.expr_clock(key.input, ProgramId::from_param(key.params[0]))
            .map_err(|e| RaptorError::invalid_parameter(e.to_string()))
    }

    /// The width of an instrument's primary bars, read from its stored bars: the most common gap
    /// between two stamps (a weekend or a holiday is a longer gap, a missing bar too). None when
    /// there are fewer than two bars, as in a live run that has seen none.
    fn primary_width_ns(&self, inst: InstrumentId) -> Option<i64> {
        if let Some(&w) = self.widths.get(&inst) {
            return Some(w);
        }
        let ts = &self.store.bars(inst)?.ts;
        let mut gaps: HashMap<i64, usize> = HashMap::new();
        for w in ts.windows(2).take(1 << 16) {
            let gap = w[1].saturating_sub(w[0]);
            if gap > 0 {
                *gaps.entry(gap).or_default() += 1;
            }
        }
        // Ties go to the shorter gap, so the answer does not depend on the hash order.
        gaps.into_iter().max_by_key(|&(gap, n)| (n, std::cmp::Reverse(gap))).map(|(gap, _)| gap)
    }

    /// The clock an expression using `at(...)` runs on when fed by `input`: a primary stream
    /// reads the run's zone and stamp and the instrument's session, and its bar width from the
    /// stored bars; a composite stream reads bars its aggregation emitted, which are stamped at
    /// the window's end, in the aggregation's own zone. None for a program with no `at(...)`.
    /// A timeframe the bars cannot be cut into is refused here, with what is valid.
    pub fn expr_clock(
        &self,
        input: IndInput,
        program: ProgramId,
    ) -> std::result::Result<Option<ClockCtx>, EvalError> {
        if !crate::runtime::exprs::info(program).needs_clock {
            return Ok(None);
        }
        let session_of = |inst: InstrumentId| {
            self.params.get(inst.index()).and_then(|p| p.spec.as_ref()?.session.clone())
        };
        let clock = match input {
            IndInput::Primary(inst) => {
                let (tz, stamp) = self.clock.ok_or(EvalError::NoClock)?;
                let mut clock = ClockCtx::new(tz, stamp, session_of(inst));
                clock.base_ns = self.primary_width_ns(inst);
                // Unknown (a live node with no bars and no width): the cell works it out from
                // its first two bars and refuses before any hook reads a value.
                clock.infer_width = clock.base_ns.is_none();
                clock
            }
            IndInput::Composite(agg) => {
                let k = self.hub.key(agg).ok_or_else(|| {
                    EvalError::Unsupported("an indicator reads an unknown aggregation".into())
                })?;
                let mut clock =
                    ClockCtx::new(k.tz_offset_ns(), BarStamp::Close, session_of(k.instrument()));
                clock.base_ns = crate::quant::fns::at::width_ns(k.spec());
                clock
            }
            IndInput::AllPrimary(_) => {
                return Err(EvalError::Unsupported(
                    "an expression reads one instrument's bars; name the instrument".into(),
                ));
            }
            // `at(...)` is refused when a cross-sectional program is interned.
            IndInput::Universe(_) => return Ok(None),
        };
        crate::runtime::exprs::check_clock(program, &clock)?;
        Ok(Some(clock))
    }

    /// Add an account for `members` strategies to share; returns its id.
    ///
    /// The account's capital and account-level limits come from `spec`; the
    /// strategies added to it keep every other setting of the run's config.
    /// `members` fixes each member's share of the starting cash, which its
    /// equity curve starts from.
    pub fn add_group(&mut self, spec: GroupSpec, members: usize) -> Result<u32> {
        if members == 0 {
            return Err(RaptorError::invalid_parameter("an account group needs a strategy"));
        }
        let gid = u32::try_from(self.groups.len())
            .map_err(|_| RaptorError::invalid_parameter("too many account groups (u32 ids)"))?;
        self.groups.push(AccountGroup::new(spec, &self.config));
        self.declared.push(members);
        Ok(gid)
    }

    /// The run's base config.
    pub fn config(&self) -> &BacktestConfig {
        &self.config
    }

    pub fn group_mut(&mut self, group: u32) -> &mut AccountGroup {
        &mut self.groups[group as usize]
    }

    pub fn store(&self) -> &MarketStore {
        &self.store
    }

    /// Add a strategy over `instruments`: non-empty, strictly ascending ids.
    /// The strategy's local instrument `i` is `instruments[i]`.
    pub fn add_strategy(&mut self, instruments: &[InstrumentId]) -> Result<StrategyId> {
        self.add_strategy_in(instruments, None)
    }

    /// [`RuntimeBuilder::add_strategy`] on the account of `group`, if given.
    pub fn add_strategy_in(
        &mut self,
        instruments: &[InstrumentId],
        group: Option<u32>,
    ) -> Result<StrategyId> {
        if instruments.is_empty() {
            return Err(RaptorError::invalid_parameter("a strategy needs an instrument"));
        }
        if instruments.windows(2).any(|w| w[0] >= w[1]) {
            return Err(RaptorError::invalid_parameter(
                "a strategy's instruments must be unique and in registration order",
            ));
        }
        if let Some(bad) = instruments.iter().find(|i| i.index() >= self.store.len()) {
            return Err(RaptorError::invalid_parameter(format!(
                "instrument id {} is not in the store",
                bad.raw()
            )));
        }
        let too_many = || RaptorError::invalid_parameter("too many strategies (u32 ids)");
        let sid = StrategyId::new(u32::try_from(self.slots.len()).map_err(|_| too_many())?);
        let inst_start = u32::try_from(self.strat_insts.len()).map_err(|_| too_many())?;
        let mut book = match group {
            None => PortfolioBook::new(self.config.clone(), self.mode),
            Some(g) => {
                let Some(target) = self.groups.get_mut(g as usize) else {
                    return Err(RaptorError::invalid_parameter("unknown account group"));
                };
                if target.members().len() >= self.declared[g as usize] {
                    return Err(RaptorError::invalid_parameter(
                        "more strategies than the account group was declared with",
                    ));
                }
                let spec = *target.spec();
                let mut book = PortfolioBook::new(spec.config(&self.config), spec.mode);
                book.join_group(spec.initial_capital / self.declared[g as usize] as f64);
                target.add_member(sid.index());
                book
            }
        };
        for &id in instruments {
            let p = &self.params[id.index()];
            let symbol = self.store.symbols().symbol(id).unwrap_or_default().to_string();
            book.add_instrument(
                symbol,
                p.direction,
                p.spec.clone(),
                p.config.as_ref(),
                self.policy,
            );
        }
        self.index.register(sid, instruments)?;
        self.strat_insts.extend_from_slice(instruments);
        self.slots.push(StrategySlot { inst_start, inst_len: instruments.len() as u32, events: 0 });
        self.books.push(Some(book));
        self.group_of.push(group);
        Ok(sid)
    }

    /// The strategy's local index of `instrument`.
    pub fn local_of(&self, strategy: StrategyId, instrument: InstrumentId) -> Option<usize> {
        local_of(&self.slots, &self.strat_insts, strategy, instrument)
    }

    /// The global instrument behind a strategy's local index.
    pub fn instrument_of(&self, strategy: StrategyId, local: usize) -> Option<InstrumentId> {
        instrument_of(&self.slots, &self.strat_insts, strategy, local)
    }

    /// Number of instruments the strategy trades.
    pub fn instruments_of(&self, strategy: StrategyId) -> usize {
        self.slots[strategy.index()].inst_len as usize
    }

    /// A strategy's book, so hooks that run before the first event
    /// (`on_start`) can query and configure it.
    pub fn book(&self, strategy: StrategyId) -> &PortfolioBook {
        self.books[strategy.index()].as_ref().expect("books are live while building")
    }

    pub fn book_mut(&mut self, strategy: StrategyId) -> &mut PortfolioBook {
        self.books[strategy.index()].as_mut().expect("books are live while building")
    }

    /// Adopt a position the account already holds; see
    /// [`PortfolioBook::adopt_position`]. In a group the cost comes out of
    /// the group's pool.
    pub fn adopt_position(
        &mut self,
        strategy: StrategyId,
        instrument: usize,
        timestamp: crate::core::types::Timestamp,
        price: f64,
        size: f64,
    ) -> std::result::Result<u64, String> {
        self.adopt_position_with(strategy, instrument, timestamp, price, size, Adoption::default())
    }

    /// [`RuntimeBuilder::adopt_position`] with a signed size and stated
    /// margin or cost; see [`PortfolioBook::adopt_position_with`].
    pub fn adopt_position_with(
        &mut self,
        strategy: StrategyId,
        instrument: usize,
        timestamp: crate::core::types::Timestamp,
        price: f64,
        size: f64,
        extras: Adoption,
    ) -> std::result::Result<u64, String> {
        let id = adopt(&mut self.groups, &self.group_of, &mut self.books, strategy, |book| {
            book.adopt_position_with(instrument, timestamp, price, size, &extras)
        })?;
        self.adopted.push((strategy, instrument, timestamp, price, size, extras));
        Ok(id)
    }

    /// Refuse a run that cannot be sent to a venue. Positions move only on
    /// orders there, so stops, targets, schedules and the session square-off
    /// are turned into orders by the runtime; what cannot be is refused here:
    /// a shared account group has one pool that no copy can follow, and a
    /// forced liquidation on a margin call is made in-process.
    pub fn check_venue_safe(&self) -> Result<()> {
        if !self.groups.is_empty() {
            return Err(RaptorError::invalid_parameter(
                "a venue-driven run cannot share an account group between strategies",
            ));
        }
        if self.config.liquidate_on_margin_call {
            return Err(RaptorError::invalid_parameter(
                "a venue-driven run cannot liquidate on a margin call: the venue does that",
            ));
        }
        Ok(())
    }

    /// Hand every strategy's protection, schedule slices and square-off to
    /// the runtime (see [`EngineKernel::set_venue_managed`]).
    ///
    /// [`EngineKernel::set_venue_managed`]: crate::portfolio::kernel::EngineKernel::set_venue_managed
    pub fn manage_books_by_venue(&mut self) {
        for (k, slot) in self.slots.iter().enumerate() {
            if let Some(book) = self.books[k].as_mut() {
                for i in 0..slot.inst_len as usize {
                    book.kernel_mut(i).set_venue_managed();
                }
            }
        }
    }

    /// A fresh book for every strategy registered so far, configured as the
    /// strategy's own is (same account, instruments, fees and adopted
    /// positions), with the symbols of its instruments in order. A venue that
    /// simulates an exchange steps these. See [`Self::check_venue_safe`] for
    /// what is refused.
    pub fn shadow_books(&self) -> Result<Vec<(Vec<String>, PortfolioBook)>> {
        self.check_venue_safe()?;
        let mut out = Vec::with_capacity(self.slots.len());
        for (k, slot) in self.slots.iter().enumerate() {
            let sid = StrategyId::new(k as u32);
            let mut book = PortfolioBook::new(self.config.clone(), self.mode);
            // Nothing reads a shadow book's curves, so keep them small.
            book.set_curve_window(Some(1));
            let mut symbols = Vec::new();
            let range = slot.inst_start as usize..(slot.inst_start + slot.inst_len) as usize;
            for &id in &self.strat_insts[range] {
                let p = &self.params[id.index()];
                let symbol = self.store.symbols().symbol(id).unwrap_or_default().to_string();
                book.add_instrument(
                    symbol.clone(),
                    p.direction,
                    p.spec.clone(),
                    p.config.as_ref(),
                    self.policy,
                );
                symbols.push(symbol);
            }
            for &(strategy, instrument, ts, price, size, extras) in &self.adopted {
                if strategy == sid {
                    book.adopt_position_with(instrument, ts, price, size, &extras)
                        .map_err(RaptorError::invalid_parameter)?;
                }
            }
            out.push((symbols, book));
        }
        Ok(out)
    }

    /// Subscribe `strategy` to an aggregation of one of its instruments.
    /// Returns the shared aggregation and the strategy's local stream id.
    pub fn register_aggregation(
        &mut self,
        strategy: StrategyId,
        key: AggKey,
    ) -> Result<(AggId, u16)> {
        if self.local_of(strategy, key.instrument()).is_none() {
            return Err(RaptorError::invalid_parameter(
                "an aggregation must be on an instrument the strategy trades",
            ));
        }
        let spec = self.params[key.instrument().index()].spec.as_ref();
        let session = spec.and_then(|s| s.session.as_ref());
        // Composites of a calendar unit cut at the session open; a venue no preset covers and
        // given no session has none, and the clock grid would be a silent guess.
        if let Some(spec) = spec.filter(|s| s.session.is_none() && s.venue.is_some()) {
            if crate::data::session_bars::uses_calendar(key.spec().unit) {
                return Err(RaptorError::invalid_parameter(format!(
                    "{:?} on venue {:?} has no session (the presets cover NSE, BSE, NFO, BFO, \
                     MCX, CDS and BCD), so its composite bars have no session grid to cut on; give the \
                     InstrumentSpec a session=",
                    spec.symbol,
                    spec.venue.as_deref().unwrap_or_default()
                )));
            }
        }
        self.hub.register_with_session(strategy, key, session)
    }

    /// Aggregate `instrument`'s trade prints into `spec` bars: the primary
    /// bars a tick run fires `on_bar` for and feeds primary indicators with.
    /// Daily and longer units cut at `tz_offset_ns`'s midnight.
    pub fn set_primary_bars(
        &mut self,
        instrument: InstrumentId,
        spec: BarSpec,
        tz_offset_ns: i64,
    ) -> Result<()> {
        self.hub.set_primary(instrument, spec, tz_offset_ns)
    }

    /// Intern the instrument set of an indicator fed by several instruments.
    pub fn primary_order(&mut self, instruments: &[InstrumentId]) -> Result<PrimaryOrder> {
        self.registry.primary_order(instruments)
    }

    /// Register an indicator reader; see [`IndicatorRegistry::register`].
    pub fn register_indicator(
        &mut self,
        strategy: StrategyId,
        key: IndKey,
        stream: Option<LocalStreamId>,
    ) -> Result<Registration> {
        self.registry.register(strategy, key, stream)
    }

    /// Freeze the registrations and build the core.
    pub fn build(self) -> Result<RuntimeCore> {
        let clocks = self
            .registry
            .keys()
            .iter()
            .map(|key| self.cell_clock(key))
            .collect::<Result<Vec<_>>>()?;
        let hub = self.hub.build()?;
        let bank = IndicatorBank::with_clocks(&self.registry, &clocks)?;
        let index = self.index.build()?;
        let mut atr_cache = AtrCache::new();
        let mut strat_atr = Vec::with_capacity(self.strat_insts.len());
        for &id in &self.strat_insts {
            let period = self.params[id.index()].atr_period;
            // `atr_at` is indexed by event ordinal, which is the bar row only
            // for an instrument with nothing but bars (see `new`).
            let series = self.store.bars(id).and_then(|bars| atr_cache.get(id, period, bars));
            strat_atr.push(series);
        }
        let instruments = self.store.len();
        Ok(RuntimeCore {
            store: self.store,
            policy: self.policy,
            hub,
            bank,
            index,
            slots: self.slots,
            strat_insts: self.strat_insts,
            strat_atr,
            books: self.books,
            groups: self.groups,
            group_of: self.group_of,
            atr_cache,
            watermark: vec![0; instruments],
            tick_watermark: vec![0; instruments],
            current: None,
            current_depth: None,
            events: 0,
            group: Vec::new(),
            carried: Vec::new(),
            selected: 0,
            spare: Vec::new(),
        })
    }
}

/// Run `f` on a strategy's book, with the group's pool if it has one, and
/// leave every member pointed at the pool afterwards.
fn adopt<R>(
    groups: &mut [AccountGroup],
    group_of: &[Option<u32>],
    books: &mut [Option<PortfolioBook>],
    strategy: StrategyId,
    f: impl FnOnce(&mut PortfolioBook) -> R,
) -> R {
    let book = books[strategy.index()].as_mut().expect("strategy book already finished");
    let Some(g) = group_of[strategy.index()] else { return f(book) };
    groups[g as usize].run_on(books, strategy.index(), f)
}

fn ohlc(bar: &OhlcvBar) -> Ohlc {
    Ohlc::from(bar)
}

fn empty_bar(timestamp: i64) -> KernelBar {
    KernelBar { timestamp, open: 0.0, high: 0.0, low: 0.0, close: 0.0, volume: 0.0 }
}

fn local_of(
    slots: &[StrategySlot],
    insts: &[InstrumentId],
    strategy: StrategyId,
    instrument: InstrumentId,
) -> Option<usize> {
    let slot = slots.get(strategy.index())?;
    let own = &insts[slot.inst_start as usize..(slot.inst_start + slot.inst_len) as usize];
    own.binary_search(&instrument).ok()
}

fn instrument_of(
    slots: &[StrategySlot],
    insts: &[InstrumentId],
    strategy: StrategyId,
    local: usize,
) -> Option<InstrumentId> {
    let slot = slots.get(strategy.index())?;
    (local < slot.inst_len as usize).then(|| insts[slot.inst_start as usize + local])
}

/// The payload of one event, whether it came off a cursor or was pushed.
#[derive(Debug, Clone, Copy)]
pub enum Pushed {
    Bar(KernelBar),
    Trade(crate::data::events::TradeTick),
    Quote(crate::data::events::QuoteTick),
    Depth(crate::data::book::DepthTick),
}

/// The event being processed.
#[derive(Debug, Clone, Copy)]
pub struct Current {
    pub event: EventRef,
    /// The bar of a bar event. For a print, its degenerate bar; for a quote
    /// or a book, zeros at the event's time. Read `data` to tell them apart.
    pub bar: KernelBar,
    /// The payload (a depth snapshot is `Depth(_)`; its levels are in the
    /// store, see [`RuntimeCore::depth_of`]).
    pub data: ScheduleData,
    /// The primary bar this print completed, if any.
    pub primary: Option<OhlcvBar>,
}

/// Heap bytes by category; see [`RuntimeCore::memory_report`].
///
/// The categories are disjoint, so `total()` is their sum. Python objects are
/// not counted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemoryReport {
    pub strategies: usize,
    /// The columns, once.
    pub shared_store: usize,
    /// Distinct aggregators and their per-instrument index (not the
    /// per-strategy subscription entries).
    pub aggregators: usize,
    /// Distinct indicator cells and cores (not the per-strategy reader
    /// entries).
    pub indicators: usize,
    /// The instrument-to-strategy index's offsets (not its member entries).
    pub index: usize,
    /// ATR series, once per instrument and period.
    pub atr: usize,
    /// Per-strategy engine state: slot, instrument list, ATR references,
    /// subscription-index entries, aggregator subscriptions, indicator reader
    /// entries.
    pub engine_per_strategy: usize,
    /// Books: the structs, their kernels, order logs, marks and trade
    /// records, curves excluded.
    pub portfolio_per_strategy: usize,
    /// Sampled equity, drawdown, returns and timestamps.
    pub curves: usize,
}

impl MemoryReport {
    pub fn total(&self) -> usize {
        self.shared_store
            + self.aggregators
            + self.indicators
            + self.index
            + self.atr
            + self.engine_per_strategy
            + self.portfolio_per_strategy
            + self.curves
    }

    fn per(&self, bytes: usize) -> usize {
        bytes / self.strategies.max(1)
    }

    pub fn engine_bytes_per_strategy(&self) -> usize {
        self.per(self.engine_per_strategy)
    }

    pub fn portfolio_bytes_per_strategy(&self) -> usize {
        self.per(self.portfolio_per_strategy)
    }

    pub fn curve_bytes_per_strategy(&self) -> usize {
        self.per(self.curves)
    }
}

/// The shared data layer and every strategy's book.
pub struct RuntimeCore {
    store: Arc<MarketStore>,
    /// How a book treats positions on an instrument added to the running core.
    policy: PositionPolicy,
    hub: AggregatorHub,
    bank: Arc<IndicatorBank>,
    index: SubscriptionIndex,
    slots: Vec<StrategySlot>,
    strat_insts: Vec<InstrumentId>,
    /// ATR series per strategy-local instrument, parallel to `strat_insts`.
    strat_atr: Vec<Option<Arc<[f64]>>>,
    /// `None` once a strategy's book has been finished.
    books: Vec<Option<PortfolioBook>>,
    groups: Vec<AccountGroup>,
    /// Each strategy's group.
    group_of: Vec<Option<u32>>,
    atr_cache: AtrCache,
    /// Bar rows visible per instrument: bar events yielded so far.
    watermark: Vec<u64>,
    /// Tick rows visible per instrument: one past the last row that yielded
    /// a trade or a quote.
    tick_watermark: Vec<u64>,
    current: Option<Current>,
    /// The snapshot of the current event when it is a book.
    current_depth: Option<crate::data::book::DepthTick>,
    events: u64,
    /// The events of the timestamp group in flight, empty between groups. See
    /// [`RuntimeCore::begin_group`].
    group: Vec<GroupSlot>,
    /// Lifecycle events of the current instant applied ahead of a bar's resting part
    /// ([`RuntimeCore::rest_event`]) and not yet handed to the strategy's `begin_event`.
    carried: Vec<(StrategyId, Vec<EngineEvent>)>,
    /// The slot of `group` that is current.
    selected: usize,
    /// Slots of finished groups, kept with their buffers for the next group.
    spare: Vec<GroupSlot>,
}

/// One event of a timestamp group: ingested, waiting to be decided and matched.
///
/// The aggregator hub and the indicator bank keep only the last event's
/// emissions, so what each subscriber is owed from this event is taken when it
/// is ingested.
struct GroupSlot {
    current: Current,
    depth: Option<crate::data::book::DepthTick>,
    /// The indicator bank's number for this event (see `IndicatorBank::event_mark`).
    bank_event: u64,
    /// Composite bars owed to each subscriber that has any.
    deliveries: Vec<(StrategyId, Vec<(u16, OhlcvBar)>)>,
    /// What the lifecycle did to each subscriber's book before the group was
    /// decided, still to be heard.
    lifecycle: Vec<(StrategyId, Vec<EngineEvent>)>,
    /// One view per subscriber of the event's instrument, in the order
    /// [`RuntimeCore::subscribers`] gives them.
    views: Vec<View>,
    /// The price the event marks its instrument at, if it prints one.
    close: Option<f64>,
}

/// What one subscriber's book holds of one event of a group.
#[derive(Clone, Copy)]
struct View {
    /// The event's instrument in the subscriber's own numbering, `NO_LOCAL`
    /// when the subscriber has no book.
    local: usize,
    /// The subscriber's mark of the event's instrument before the group, so
    /// a decision on the event reads its own instrument as it would alone.
    before: Option<f64>,
    /// The book is its own account's and has no lifecycle to run: it was
    /// marked when the event was ingested, and nothing more is done to it
    /// before the group is decided.
    plain: bool,
}

const NO_LOCAL: usize = usize::MAX;

impl RuntimeCore {
    /// The market data. Clone the `Arc` and build the cursor from the clone:
    /// `MergeCursor::new(&store)`.
    pub fn store(&self) -> &Arc<MarketStore> {
        &self.store
    }

    pub fn bank(&self) -> &Arc<IndicatorBank> {
        &self.bank
    }

    pub fn hub(&self) -> &AggregatorHub {
        &self.hub
    }

    pub fn strategies(&self) -> usize {
        self.slots.len()
    }

    /// Events advanced so far.
    pub fn events(&self) -> u64 {
        self.events
    }

    /// Take the next event off the cursor and make it current.
    ///
    /// Moves the visibility watermark, pushes the event through the shared
    /// aggregators and feeds the indicator bank: `begin_event`, then every
    /// composite emission in emission order, then the primary bar. For a
    /// print the primary bar is the one the print completes, if any; quotes
    /// and books feed nothing. Strategy code runs after this returns.
    /// `Ok(None)` at the end of the tape.
    pub fn advance(&mut self, cursor: &mut MergeCursor<'_>) -> Result<Option<EventRef>> {
        let Some((event, payload)) = Self::pull(cursor)? else {
            self.current = None;
            return Ok(None);
        };
        self.ingest(event, payload)?;
        Ok(Some(event))
    }

    /// Take the next event and its payload off the cursor without ingesting
    /// it: the events of a timestamp group are pulled first, then ingested
    /// together by [`RuntimeCore::begin_group`].
    pub fn pull(cursor: &mut MergeCursor<'_>) -> Result<Option<(EventRef, Pushed)>> {
        let Some(event) = cursor.next() else { return Ok(None) };
        let missing = || RaptorError::invalid_parameter("the cursor yielded an event with no row");
        let payload = match event.kind {
            EventKind::Bar => Pushed::Bar(cursor.bar(&event).ok_or_else(missing)?),
            EventKind::Trade => Pushed::Trade(cursor.trade(&event).ok_or_else(missing)?),
            EventKind::Quote => Pushed::Quote(cursor.quote(&event).ok_or_else(missing)?),
            EventKind::Depth => Pushed::Depth(cursor.depth(&event).ok_or_else(missing)?),
        };
        Ok(Some((event, payload)))
    }

    /// Make a pushed event current, as [`RuntimeCore::advance`] does for the
    /// cursor's: the same watermark, aggregator and indicator work, from a
    /// payload the caller already holds. `event.row` is the absolute row of
    /// the payload in the caller's ring.
    pub fn advance_pushed(&mut self, event: EventRef, payload: Pushed) -> Result<()> {
        self.ingest(event, payload)
    }

    fn ingest(&mut self, event: EventRef, payload: Pushed) -> Result<()> {
        let cur = self.ingest_event(event, payload)?;
        // A lone event is a date of its own: cross-sectional expressions step on it now.
        self.bank.seal();
        self.current = Some(cur);
        Ok(())
    }

    /// The work of taking an event in. A book event's snapshot is left in
    /// `current_depth`.
    fn ingest_event(&mut self, event: EventRef, payload: Pushed) -> Result<Current> {
        let i = event.instrument.index();
        self.events += 1;
        self.bank.begin_event();
        let mut primary = None;
        self.current_depth = None;
        let (bar, data) = match payload {
            Pushed::Bar(bar) => {
                self.watermark[i] = event.row + 1;
                for (agg, composite) in self.hub.on_bar(event.instrument, &bar) {
                    self.bank.emit_composite(*agg, ohlc(composite))?;
                }
                self.bank.emit_primary(event.instrument, Ohlc::from(&bar))?;
                (bar, ScheduleData::Bar(bar))
            }
            Pushed::Trade(t) => {
                self.tick_watermark[i] = self.tick_watermark[i].max(event.row + 1);
                for (agg, composite) in
                    self.hub.on_trade(event.instrument, t.timestamp, t.price, t.size)
                {
                    self.bank.emit_composite(*agg, ohlc(composite))?;
                }
                primary = self.hub.primary_trade(event.instrument, t.timestamp, t.price, t.size);
                if let Some(bar) = &primary {
                    self.bank.emit_primary(event.instrument, ohlc(bar))?;
                }
                let bar = KernelBar {
                    timestamp: t.timestamp,
                    open: t.price,
                    high: t.price,
                    low: t.price,
                    close: t.price,
                    volume: t.size,
                };
                (bar, ScheduleData::Trade(t))
            }
            Pushed::Quote(q) => {
                self.tick_watermark[i] = self.tick_watermark[i].max(event.row + 1);
                self.hub.skip(event.instrument);
                (empty_bar(q.timestamp), ScheduleData::Quote(q))
            }
            Pushed::Depth(d) => {
                self.hub.skip(event.instrument);
                self.current_depth = Some(d);
                (
                    empty_bar(event.ts),
                    // The slot is a store row; a pushed snapshot travels in `depth`.
                    ScheduleData::Depth(DepthRef { slot: event.row as u32, timestamp: event.ts }),
                )
            }
        };
        Ok(Current { event, bar, data, primary })
    }

    /// Ingest every event of one timestamp group (see
    /// `docs/architecture/event-semantics.md`, the two-phase rule), apply what
    /// the books must have applied before anything is decided, and leave the
    /// first event selected.
    ///
    /// 1. Each event is ingested: the visibility watermark, the aggregators and
    ///    the indicator bank, in the order given. Afterwards every
    ///    instrument's event of the group is visible to every strategy.
    /// 2. Each subscriber's lifecycle runs (corporate actions, dividends,
    ///    expiries), for each event in turn; the events it produced are heard
    ///    by [`RuntimeCore::begin_event`] when that event is decided.
    /// 3. Each subscriber's book marks the instrument at the event's price, so
    ///    equity at a decision does not depend on the order of the group.
    ///
    /// Then, for each event in order, [`RuntimeCore::select`] it, run the
    /// strategies' handlers, and later [`RuntimeCore::select`] it again to
    /// [`RuntimeCore::apply`] it. [`RuntimeCore::end_group`] closes the group.
    ///
    /// The caller gives each instrument at most once.
    pub fn begin_group(&mut self, events: &[(EventRef, Pushed)]) -> Result<()> {
        if !self.group.is_empty() {
            return Err(RaptorError::invalid_parameter("a timestamp group is already open"));
        }
        let opened = self.ingest_group(events);
        if opened.is_err() {
            self.end_group();
        }
        opened
    }

    fn ingest_group(&mut self, events: &[(EventRef, Pushed)]) -> Result<()> {
        self.bank.open_group();
        for &(event, payload) in events {
            let current = self.ingest_event(event, payload)?;
            let depth = self.current_depth.take();
            let mut slot = self.spare.pop().unwrap_or_else(|| GroupSlot {
                current,
                depth: None,
                bank_event: 0,
                deliveries: Vec::new(),
                lifecycle: Vec::new(),
                views: Vec::new(),
                close: None,
            });
            slot.current = current;
            slot.depth = depth;
            slot.bank_event = self.bank.event_mark();
            slot.close = match current.data {
                ScheduleData::Bar(b) => Some(b.close),
                ScheduleData::Trade(t) => Some(t.price),
                _ => None,
            };
            if !self.hub.emissions().is_empty() {
                for &sid in self.index.subscribers(event.instrument) {
                    let owed: Vec<_> = self.hub.deliveries(sid).map(|(l, b)| (l, *b)).collect();
                    if !owed.is_empty() {
                        slot.deliveries.push((sid, owed));
                    }
                }
            }
            // The books learn the event is in the group, and each subscriber's
            // own number for the instrument is looked up once. A book that is
            // its own account's and has no lifecycle to run is marked now:
            // the later rounds would find nothing to do to it but the mark.
            let entry = |local| ScheduleEntry {
                instrument: local,
                local_idx: event.local_idx as usize,
                data: current.data,
            };
            for k in 0..self.index.subscribers(event.instrument).len() {
                let sid = self.index.subscribers(event.instrument)[k];
                let local = local_of(&self.slots, &self.strat_insts, sid, event.instrument);
                let alone = self.group_of[sid.index()].is_none();
                let view = match (local, self.books[sid.index()].as_mut()) {
                    (Some(local), Some(book)) => {
                        book.group_enter(local);
                        if alone && !book.lifecycle_active() {
                            let before = book.premark(&entry(local));
                            View { local, before, plain: true }
                        } else {
                            View { local, before: None, plain: false }
                        }
                    }
                    _ => View { local: NO_LOCAL, before: None, plain: false },
                };
                slot.views.push(view);
            }
            self.group.push(slot);
        }
        // The whole group is one date: cross-sectional expressions step once, on every bar of
        // it, before any strategy reveals anything.
        self.bank.seal();
        for i in 0..self.group.len() {
            self.select(i);
            let instrument = self.group[i].current.event.instrument;
            for k in 0..self.index.subscribers(instrument).len() {
                if self.group[i].views[k].plain {
                    continue;
                }
                let sid = self.index.subscribers(instrument)[k];
                let events = self.lifecycle_of_current(sid)?;
                if !events.is_empty() {
                    self.group[i].lifecycle.push((sid, events));
                }
            }
        }
        for i in 0..self.group.len() {
            let cur = self.group[i].current;
            for k in 0..self.group[i].views.len() {
                let View { local, plain, .. } = self.group[i].views[k];
                if local == NO_LOCAL || plain {
                    continue;
                }
                let sid = self.index.subscribers(cur.event.instrument)[k];
                let entry = ScheduleEntry {
                    instrument: local,
                    local_idx: cur.event.local_idx as usize,
                    data: cur.data,
                };
                if let Some(book) = self.books[sid.index()].as_mut() {
                    self.group[i].views[k].before = book.premark(&entry);
                }
            }
        }
        self.select(0);
        Ok(())
    }

    /// Make event `i` of the open group current: what the handlers and the
    /// books see of the event in flight, and the indicator bank's view of it.
    pub fn select(&mut self, i: usize) {
        let slot = &self.group[i];
        self.selected = i;
        self.current = Some(slot.current);
        self.current_depth = slot.depth;
        self.bank.set_event(slot.bank_event);
    }

    /// Point `strategy`'s book (the `at`-th subscriber of the selected event's
    /// instrument) at the view of the selected event's decision:
    /// the other events of the group, as far as it subscribes to them, are
    /// marked at the group's prices, and the event's own instrument at the
    /// mark it had before the group (`own_before`), as a lone event is decided.
    /// Call with `false` once the strategy has decided, to leave the group
    /// marked everywhere.
    pub fn decide_view(&mut self, strategy: StrategyId, at: usize, own_before: bool) {
        let Some(slot) = self.group.get(self.selected) else { return };
        // `at` is the strategy's place among the subscribers of the event's
        // instrument, which is the place of its view.
        let View { local, before, .. } = slot.views[at];
        debug_assert_eq!(self.index.subscribers(slot.current.event.instrument)[at], strategy);
        if local == NO_LOCAL {
            return;
        }
        let mark = if own_before { before } else { slot.close.or(before) };
        if let Some(book) = self.books[strategy.index()].as_mut() {
            book.set_mark(local, mark);
        }
    }

    /// Events in the open group, `0` between groups.
    pub fn group_len(&self) -> usize {
        self.group.len()
    }

    /// Whether a timestamp group is open.
    pub fn grouped(&self) -> bool {
        !self.group.is_empty()
    }

    /// Event `i` of the open group.
    pub fn group_event(&self, i: usize) -> &Current {
        &self.group[i].current
    }

    /// The most recent bar a group holds for `instrument`, if it has one: what
    /// a decision at the group's timestamp sees of that instrument.
    pub fn group_bar(&self, instrument: InstrumentId) -> Option<KernelBar> {
        self.group
            .iter()
            .find(|s| s.current.event.instrument == instrument)
            .map(|s| s.current.bar)
            .filter(|bar| bar.close > 0.0)
    }

    /// Close the open group. The indicator bank is left on the group's last
    /// event, so the next event is numbered past every stamp.
    pub fn end_group(&mut self) {
        self.bank.close_group();
        if let Some(last) = self.group.last() {
            self.bank.set_event(last.bank_event);
        }
        for i in 0..self.group.len() {
            let instrument = self.group[i].current.event.instrument;
            for k in 0..self.index.subscribers(instrument).len() {
                let sid = self.index.subscribers(instrument)[k];
                if let Some(book) = self.books[sid.index()].as_mut() {
                    book.group_leave();
                }
            }
        }
        for mut slot in self.group.drain(..) {
            slot.deliveries.clear();
            slot.lifecycle.clear();
            slot.views.clear();
            self.spare.push(slot);
        }
        self.selected = 0;
    }

    /// What `strategy` is owed from the aggregators for the current event:
    /// its own subscriptions on that instrument, in subscription order, as
    /// `(local_stream, bar)`. In a group these are the ones taken when the
    /// event was ingested.
    pub fn deliveries_into(&self, strategy: StrategyId, out: &mut Vec<(u16, OhlcvBar)>) {
        out.clear();
        if let Some(slot) = self.group.get(self.selected) {
            if let Some((_, owed)) = slot.deliveries.iter().find(|(sid, _)| *sid == strategy) {
                out.extend_from_slice(owed);
            }
            return;
        }
        out.extend(self.hub.deliveries(strategy).map(|(l, b)| (l, *b)));
    }

    /// Forget the current event, so nothing is in flight. A live driver calls
    /// this once a push has been fully dispatched: work queued afterwards
    /// (a time event on a refused row) is stamped on its target's last print
    /// rather than on the event just finished.
    pub fn clear_current(&mut self) {
        self.current = None;
        self.current_depth = None;
    }

    pub fn current(&self) -> Option<&Current> {
        self.current.as_ref()
    }

    /// Strategies subscribed to `instrument`, in registration order.
    #[inline]
    pub fn subscribers(&self, instrument: InstrumentId) -> &[StrategyId] {
        self.index.subscribers(instrument)
    }

    /// What `strategy` receives from the aggregators for the current event.
    pub fn deliveries(&self, strategy: StrategyId) -> crate::runtime::aggregators::Deliveries<'_> {
        self.hub.deliveries(strategy)
    }

    /// Bar rows of `instrument` visible so far.
    #[inline]
    pub fn watermark(&self, instrument: InstrumentId) -> u64 {
        self.watermark.get(instrument.index()).copied().unwrap_or(0)
    }

    /// Tick rows of `instrument` visible so far: one past the last row that
    /// has yielded a trade or a quote.
    #[inline]
    pub fn tick_watermark(&self, instrument: InstrumentId) -> u64 {
        self.tick_watermark.get(instrument.index()).copied().unwrap_or(0)
    }

    /// The depth snapshot behind the current event, if it is a book.
    pub fn depth_of(&self, event: &EventRef) -> Option<crate::data::book::DepthTick> {
        (event.kind == EventKind::Depth).then_some(self.current_depth).flatten()
    }

    pub fn local_of(&self, strategy: StrategyId, instrument: InstrumentId) -> Option<usize> {
        local_of(&self.slots, &self.strat_insts, strategy, instrument)
    }

    pub fn instrument_of(&self, strategy: StrategyId, local: usize) -> Option<InstrumentId> {
        instrument_of(&self.slots, &self.strat_insts, strategy, local)
    }

    /// Number of instruments the strategy trades.
    pub fn instruments_of(&self, strategy: StrategyId) -> usize {
        self.slots[strategy.index()].inst_len as usize
    }

    /// The ATR for a bar of a strategy-local instrument; `0.0` (arms
    /// nothing) without a series, as the legacy runner.
    #[inline]
    pub fn atr_at(&self, strategy: StrategyId, local: usize, idx: usize) -> f64 {
        let slot = &self.slots[strategy.index()];
        match &self.strat_atr[slot.inst_start as usize + local] {
            Some(series) => series[idx],
            None => 0.0,
        }
    }

    /// Add an instrument to the running core, as the last (highest) id, and subscribe the
    /// `strategies` to it. Each strategy's book gets it as its newest local instrument, so no
    /// existing local index, order id or event ordinal moves. `primary` cuts the instrument's
    /// primary bars from trade prints, as at build. `venue_managed` is what
    /// [`RuntimeBuilder::manage_books_by_venue`] did for the instruments of a node whose orders
    /// go to a venue.
    ///
    /// Nothing is registered when an argument is refused.
    pub fn add_instrument(
        &mut self,
        symbol: &str,
        params: &InstrumentParams,
        strategies: &[StrategyId],
        primary: Option<(BarSpec, i64)>,
        venue_managed: bool,
    ) -> Result<InstrumentId> {
        if self.store.symbols().get(symbol).is_some() {
            return Err(RaptorError::invalid_parameter(format!(
                "{symbol} is already an instrument of this run"
            )));
        }
        if strategies.windows(2).any(|w| w[0] >= w[1]) {
            return Err(RaptorError::invalid_parameter(
                "an instrument's strategies must be unique and in registration order",
            ));
        }
        for sid in strategies {
            if self.books.get(sid.index()).is_none_or(Option::is_none) {
                return Err(RaptorError::invalid_parameter(format!(
                    "strategy {} is unknown or finished",
                    sid.index()
                )));
            }
        }
        if params.atr_period > 0 {
            return Err(RaptorError::invalid_parameter(
                "ATR-based stop/target on an instrument added to a running node: tick events \
                 carry no bar series to compute ATR from",
            ));
        }
        let mut store = (*self.store).clone();
        let id = store.add_instrument(symbol)?;
        debug_assert_eq!(id.index(), self.watermark.len());
        self.hub.add_instrument(primary)?;
        self.index.set_subscribers(id, strategies)?;
        self.store = Arc::new(store);
        self.watermark.push(0);
        self.tick_watermark.push(0);
        // Re-lay the strategies' contiguous slices: each strategy that takes the instrument
        // gets it after its own, which keeps its ids ascending.
        let mut insts = Vec::with_capacity(self.strat_insts.len() + strategies.len());
        let mut atr = Vec::with_capacity(insts.capacity());
        for (k, slot) in self.slots.iter_mut().enumerate() {
            let range = slot.inst_start as usize..(slot.inst_start + slot.inst_len) as usize;
            slot.inst_start = insts.len() as u32;
            insts.extend_from_slice(&self.strat_insts[range.clone()]);
            atr.extend(self.strat_atr[range].iter().cloned());
            if strategies.contains(&StrategyId::new(k as u32)) {
                insts.push(id);
                atr.push(None);
                slot.inst_len += 1;
                let book = self.books[k].as_mut().expect("checked above");
                book.add_instrument(
                    symbol.to_string(),
                    params.direction,
                    params.spec.clone(),
                    params.config.as_ref(),
                    self.policy,
                );
                if venue_managed {
                    book.kernel_mut(slot.inst_len as usize - 1).set_venue_managed();
                }
            }
        }
        self.strat_insts = insts;
        self.strat_atr = atr;
        Ok(id)
    }

    /// Subscribe `strategy` to an aggregation of the newest instrument; see
    /// [`AggregatorHub::subscribe`].
    pub fn subscribe_aggregation(
        &mut self,
        strategy: StrategyId,
        key: AggKey,
        session: Option<&crate::core::session_schedule::SessionSchedule>,
    ) -> Result<(AggId, u16)> {
        self.hub.subscribe(strategy, key, session)
    }

    /// What keeps `instrument` in use: per subscribed strategy, `(strategy, open positions,
    /// working orders)` for those that hold any. Empty means it can be retired.
    pub fn instrument_in_use(&self, instrument: InstrumentId) -> Vec<(StrategyId, usize, usize)> {
        self.index
            .subscribers(instrument)
            .iter()
            .filter_map(|&sid| {
                let local = self.local_of(sid, instrument)?;
                let kernel = self.books.get(sid.index())?.as_ref()?.kernel(local);
                let (positions, orders) = (kernel.open_count(), kernel.open_orders().len());
                (positions + orders > 0).then_some((sid, positions, orders))
            })
            .collect()
    }

    /// Stop the core feeding `instrument`: no strategy is subscribed to it any more and its
    /// aggregators are freed. Its id, and each book's kernel for it (which holds its trades and
    /// order history for the results), stay. The caller has checked
    /// [`RuntimeCore::instrument_in_use`].
    pub fn retire_instrument(&mut self, instrument: InstrumentId) -> Result<()> {
        let holders = self.holders(instrument);
        self.index.set_subscribers(instrument, &[])?;
        self.hub.retire_instrument(instrument)?;
        for sid in holders {
            let local = self.local_of(sid, instrument).expect("a holder");
            if let Some(book) = self.books[sid.index()].as_mut() {
                book.retire_kernel(local).map_err(RaptorError::invalid_parameter)?;
            }
        }
        Ok(())
    }

    /// The strategies that have a book entry for `instrument`, retired or not.
    pub fn holders(&self, instrument: InstrumentId) -> Vec<StrategyId> {
        (0..self.slots.len())
            .map(|k| StrategyId::new(k as u32))
            .filter(|&sid| self.local_of(sid, instrument).is_some())
            .collect()
    }

    /// Bring a retired instrument back for `strategies` (a subset of its holders): the same id,
    /// the same books' kernels (flat, as they were retired), fresh aggregators. The caller builds
    /// fresh rings.
    pub fn revive_instrument(
        &mut self,
        instrument: InstrumentId,
        strategies: &[StrategyId],
        primary: Option<(BarSpec, i64)>,
        session: Option<&crate::core::session_schedule::SessionSchedule>,
    ) -> Result<()> {
        let holders = self.holders(instrument);
        if strategies.is_empty()
            || strategies.windows(2).any(|w| w[0] >= w[1])
            || strategies.iter().any(|s| !holders.contains(s))
        {
            return Err(RaptorError::invalid_parameter(
                "a revived instrument's strategies must be some of those that held it, each once",
            ));
        }
        if !self.index.subscribers(instrument).is_empty() {
            return Err(RaptorError::invalid_parameter("the instrument is not retired"));
        }
        self.hub.revive_instrument(instrument, primary, session)?;
        self.index.set_subscribers(instrument, strategies)?;
        for &sid in strategies {
            let local = self.local_of(sid, instrument).expect("a holder");
            if let Some(book) = self.books[sid.index()].as_mut() {
                book.revive_kernel(local);
            }
        }
        Ok(())
    }

    /// The strategy's book, or `None` once it has been finished.
    pub fn book(&self, strategy: StrategyId) -> Option<&PortfolioBook> {
        self.books.get(strategy.index())?.as_ref()
    }

    pub fn book_mut(&mut self, strategy: StrategyId) -> Option<&mut PortfolioBook> {
        self.books.get_mut(strategy.index())?.as_mut()
    }

    /// Run the strategy's lifecycle for the current event (corporate actions
    /// effective by now, dividends due, contracts expired with no bar of
    /// their own) and return its events. Call once per event per subscribed
    /// strategy, before its hooks, so the strategy sees the position after
    /// the action. [`RuntimeCore::apply`] finds nothing left to do.
    pub fn begin_event(&mut self, strategy: StrategyId) -> Result<Vec<EngineEvent>> {
        // In a group the lifecycle ran before any event of it was decided.
        if let Some(slot) = self.group.get_mut(self.selected) {
            return Ok(match slot.lifecycle.iter().position(|(sid, _)| *sid == strategy) {
                Some(at) => slot.lifecycle.remove(at).1,
                None => Vec::new(),
            });
        }
        let mut events = match self.carried.iter().position(|(sid, _)| *sid == strategy) {
            Some(at) => self.carried.remove(at).1,
            None => Vec::new(),
        };
        events.extend(self.lifecycle_of_current(strategy)?);
        Ok(events)
    }

    /// Under `next_bar_open`, meet the open of the current bar for `strategy` before it has seen
    /// the bar: the lifecycle of the instant (corporate actions, expiries) and then the fills at
    /// this open of what it decided earlier, as events for its hooks. Empty when the bar is not
    /// one the strategy's book can open ahead of the strategy (another fill timing, a halt waiting), in which case [`RuntimeCore::apply`] does it all.
    /// [`RuntimeCore::apply`] carries on from the open, and `begin_event` finds nothing left.
    pub fn open_event(&mut self, strategy: StrategyId) -> Result<Vec<EngineEvent>> {
        let Some(cur) = self.current else { return Ok(Vec::new()) };
        if !matches!(cur.data, ScheduleData::Bar(_)) {
            return Ok(Vec::new());
        }
        let Some(local) = local_of(&self.slots, &self.strat_insts, strategy, cur.event.instrument)
        else {
            return Ok(Vec::new());
        };
        let entry = ScheduleEntry {
            instrument: local,
            local_idx: cur.event.local_idx as usize,
            data: cur.data,
        };
        if !self.books[strategy.index()].as_ref().is_some_and(|b| b.can_open(&entry)) {
            return Ok(Vec::new());
        }
        let mut events = self.begin_event(strategy)?;
        if let Some(g) = self.group_of[strategy.index()] {
            events.extend(self.groups[g as usize].open(&mut self.books, strategy.index(), &entry));
            return Ok(events);
        }
        let book = self.books[strategy.index()].as_mut().expect("checked above");
        events.extend(book.open_bar(&entry));
        Ok(events)
    }

    /// Meet what rests for the current bar for `strategy` before it has decided on the bar: its
    /// stops, targets and resting orders trade the bar's range, as they do at a venue while the
    /// bar forms, and the events are returned for its hooks to hear before `on_bar`. Empty when
    /// nothing rests or the bar is not one the book can rest ahead of the strategy (a halt
    /// waiting), in which case [`RuntimeCore::apply`] does it all. The lifecycle of the instant
    /// has run first (`begin_event`), and [`RuntimeCore::apply`] carries on from here with the
    /// decisions of the bar.
    pub fn rest_event(&mut self, strategy: StrategyId) -> Result<Vec<EngineEvent>> {
        let Some(cur) = self.current else { return Ok(Vec::new()) };
        if !matches!(cur.data, ScheduleData::Bar(_)) {
            return Ok(Vec::new());
        }
        let Some(local) = local_of(&self.slots, &self.strat_insts, strategy, cur.event.instrument)
        else {
            return Ok(Vec::new());
        };
        let entry = ScheduleEntry {
            instrument: local,
            local_idx: cur.event.local_idx as usize,
            data: cur.data,
        };
        if !self.books[strategy.index()].as_ref().is_some_and(|b| b.can_rest(&entry)) {
            return Ok(Vec::new());
        }
        // The lifecycle of the instant (a corporate action, an expiry) must be applied before the
        // bar's range is traded. Its events are not this step's to hear: they are kept for
        // `begin_event`, which the strategy's event delivers first, as it always did. (In a
        // timestamp group it ran in the apply phase and its events are kept already.)
        if self.group.get(self.selected).is_none() {
            let lifecycle = self.lifecycle_of_current(strategy)?;
            if !lifecycle.is_empty() {
                self.carried.push((strategy, lifecycle));
            }
        }
        // A fill inside the bar is judged by the marks of the bars before the group: the closes of
        // the instant's other bars are not known yet (an exposure limit reads them).
        self.mark_group_before(strategy, true);
        let events = if let Some(g) = self.group_of[strategy.index()] {
            self.groups[g as usize].rest(&mut self.books, strategy.index(), &entry)
        } else {
            let book = self.books[strategy.index()].as_mut().expect("checked above");
            book.rest_bar(&entry)
        };
        self.mark_group_before(strategy, false);
        Ok(events)
    }

    /// In an open group, mark every instrument of the group that `strategy` holds at the mark it
    /// had before the group (`before`), or at the group's price again.
    fn mark_group_before(&mut self, strategy: StrategyId, before: bool) {
        for slot in &self.group {
            let instrument = slot.current.event.instrument;
            let Some(k) = self.index.subscribers(instrument).iter().position(|s| *s == strategy)
            else {
                continue;
            };
            let View { local, before: mark, .. } = slot.views[k];
            if local == NO_LOCAL {
                continue;
            }
            let mark = if before { mark } else { slot.close.or(mark) };
            if let Some(book) = self.books[strategy.index()].as_mut() {
                book.set_mark(local, mark);
            }
        }
    }

    fn lifecycle_of_current(&mut self, strategy: StrategyId) -> Result<Vec<EngineEvent>> {
        // The common case, a book with no expiry and no action to apply,
        // costs one flag read per event.
        if self.group_of[strategy.index()].is_none()
            && self.books[strategy.index()].as_ref().is_some_and(|b| !b.lifecycle_active())
        {
            return Ok(Vec::new());
        }
        let Some(cur) = self.current else { return Ok(Vec::new()) };
        let local = local_of(&self.slots, &self.strat_insts, strategy, cur.event.instrument)
            .ok_or_else(|| {
                RaptorError::invalid_parameter("strategy is not subscribed to the current event")
            })?;
        let ts = ScheduleEntry { instrument: local, local_idx: 0, data: cur.data }.timestamp();
        if let Some(g) = self.group_of[strategy.index()] {
            return Ok(self.groups[g as usize].begin_event(
                &mut self.books,
                strategy.index(),
                ts,
                local,
            ));
        }
        let book = self.books[strategy.index()].as_mut().expect("strategy book already finished");
        Ok(book.begin_event(ts, local))
    }

    /// Step the strategy's book with the current event.
    ///
    /// Call once per event per subscribed strategy, after its hooks ran.
    pub fn apply(&mut self, strategy: StrategyId, input: StepInput) -> Result<Vec<EngineEvent>> {
        let Some(cur) = self.current else {
            return Err(RaptorError::invalid_parameter("no current event to apply"));
        };
        let local = local_of(&self.slots, &self.strat_insts, strategy, cur.event.instrument)
            .ok_or_else(|| {
                RaptorError::invalid_parameter("strategy is not subscribed to the current event")
            })?;
        let entry = ScheduleEntry {
            instrument: local,
            local_idx: cur.event.local_idx as usize,
            data: cur.data,
        };
        let depth = self.depth_of(&cur.event);
        self.slots[strategy.index()].events += 1;
        if let Some(g) = self.group_of[strategy.index()] {
            return Ok(self.groups[g as usize].step(
                &mut self.books,
                strategy.index(),
                &entry,
                depth,
                input,
            ));
        }
        let book = self.books[strategy.index()].as_mut().expect("strategy book already finished");
        Ok(book.apply(&entry, depth, input))
    }

    /// The strategy's answer to a fill, a market order marked `follow_up`, met on the bar its
    /// book stepped at `ts`; in an account group the account-level margin work follows, as after a
    /// step. See [`PortfolioBook::fill_follow_up`].
    pub fn fill_follow_up(
        &mut self,
        strategy: StrategyId,
        local: usize,
        order_id: u64,
        ts: crate::core::types::Timestamp,
    ) -> Vec<EngineEvent> {
        if let Some(g) = self.group_of(strategy) {
            return self.groups[g as usize].fill_follow_up(
                &mut self.books,
                strategy.index(),
                local,
                order_id,
                ts,
            );
        }
        match self.books[strategy.index()].as_mut() {
            Some(book) => book.fill_follow_up(local, order_id, ts),
            None => Vec::new(),
        }
    }

    /// The account group of a strategy, if it is in one.
    pub fn group_of(&self, strategy: StrategyId) -> Option<u32> {
        self.group_of.get(strategy.index()).copied().flatten()
    }

    /// Halt an account by hand; see [`AccountGroup::halt`].
    pub fn halt_group(&mut self, group: u32, reason: &str, flatten: bool) -> Result<()> {
        let g = self
            .groups
            .get_mut(group as usize)
            .ok_or_else(|| RaptorError::invalid_parameter("unknown account group"))?;
        g.halt(&mut self.books, reason, flatten);
        Ok(())
    }

    /// Adopt a position; see [`RuntimeBuilder::adopt_position`].
    pub fn adopt_position(
        &mut self,
        strategy: StrategyId,
        instrument: usize,
        timestamp: crate::core::types::Timestamp,
        price: f64,
        size: f64,
    ) -> std::result::Result<u64, String> {
        self.adopt_position_with(strategy, instrument, timestamp, price, size, Adoption::default())
    }

    /// Adopt a position with a signed size and stated margin or cost; see
    /// [`RuntimeBuilder::adopt_position_with`].
    pub fn adopt_position_with(
        &mut self,
        strategy: StrategyId,
        instrument: usize,
        timestamp: crate::core::types::Timestamp,
        price: f64,
        size: f64,
        extras: Adoption,
    ) -> std::result::Result<u64, String> {
        if self.books.get(strategy.index()).is_none_or(|b| b.is_none()) {
            return Err("strategy already finished".to_string());
        }
        adopt(&mut self.groups, &self.group_of, &mut self.books, strategy, |book| {
            book.adopt_position_with(instrument, timestamp, price, size, &extras)
        })
    }

    /// An account's result, once every strategy on it is finished.
    pub fn finish_group(&mut self, group: u32) -> Result<GroupOutcome> {
        let g = self
            .groups
            .get_mut(group as usize)
            .ok_or_else(|| RaptorError::invalid_parameter("unknown account group"))?;
        if g.members().iter().any(|&m| self.books[m].is_some()) {
            return Err(RaptorError::invalid_parameter(
                "finish every strategy on the account before the account",
            ));
        }
        Ok(g.outcome())
    }

    /// Liquidate and settle the strategy's book.
    pub fn finish(&mut self, strategy: StrategyId) -> Result<SessionOutcome> {
        if let Some(g) = self.group_of[strategy.index()] {
            if self.books[strategy.index()].is_none() {
                return Err(RaptorError::invalid_parameter("strategy already finished"));
            }
            // The closes pay into the account's pool, not the book's copy.
            self.groups[g as usize].close_out(&mut self.books, strategy.index());
        }
        self.books[strategy.index()]
            .take()
            .map(PortfolioBook::finish)
            .ok_or_else(|| RaptorError::invalid_parameter("strategy already finished"))
    }

    /// Heap bytes by category (unfinished books only).
    pub fn memory_report(&self) -> MemoryReport {
        use std::mem::size_of;
        let live = || self.books.iter().flatten();
        let atr_refs = self.strat_atr.capacity() * size_of::<Option<Arc<[f64]>>>();
        let index_entries = self.index.len() * size_of::<StrategyId>();
        let hub_entries = self.hub.subscription_heap_bytes();
        let link_entries = self.bank.link_heap_bytes();
        MemoryReport {
            strategies: self.slots.len(),
            shared_store: self.store.heap_bytes(),
            aggregators: self.hub.heap_bytes().saturating_sub(hub_entries),
            indicators: self.bank.heap_bytes().saturating_sub(link_entries),
            index: self.index.heap_bytes().saturating_sub(index_entries),
            atr: self.atr_cache.heap_bytes(),
            engine_per_strategy: self.slots.capacity() * size_of::<StrategySlot>()
                + self.strat_insts.capacity() * size_of::<InstrumentId>()
                + atr_refs
                + index_entries
                + hub_entries
                + link_entries,
            // The book structs themselves sit inline in `books`.
            portfolio_per_strategy: live().map(|b| b.heap_bytes()).sum::<usize>()
                + self.books.capacity() * size_of::<Option<PortfolioBook>>(),
            curves: live().map(|b| b.curve_heap_bytes()).sum::<usize>()
                + self.groups.iter().map(|g| g.curve_heap_bytes()).sum::<usize>(),
        }
    }
}

#[cfg(test)]
#[path = "runner_tests.rs"]
mod tests;
