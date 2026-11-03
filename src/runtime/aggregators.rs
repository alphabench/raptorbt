//! One aggregator per distinct aggregation, fanned out to every subscriber.
//!
//! Strategies used to build a private bar builder per (symbol, subscription),
//! so N strategies on the same 5-minute bars repeated the same work N times.
//! The hub registers each distinct [`AggKey`] once, pushes every primary bar
//! into it once, and lets each strategy read the emissions in its own
//! subscription order.
//!
//! Per primary bar a strategy sees what a private builder would have shown it:
//! for each of its subscriptions on that instrument, in subscription order,
//! the bar the builder completed followed by any Renko bricks held back
//! behind it. Nothing is flushed at end of data, as in the runners today: the
//! in-progress bar of every aggregator is dropped.
//!
//! # Trades
//!
//! A trade print feeds the same aggregators through [`AggregatorHub::on_trade`]
//! as `BarAggregator.push_trade` would: a print is an instant, so it is pushed
//! unstamped, with no buy/sell split (`signed_volume` 0). The optional
//! *primary* bars of a tick run (the view that fires `on_bar` and feeds
//! indicators registered without a stream) are one builder per instrument,
//! set with [`AggregatorHubBuilder::set_primary`] and fed by
//! [`AggregatorHub::primary_trade`], which, like a tick run, returns only the
//! first bar a print completes.

use std::collections::HashMap;

use crate::core::error::RaptorError;
use crate::core::session_schedule::SessionSchedule;
use crate::core::types::OhlcvBar;
use crate::core::Result;
use crate::data::{
    builder_for_session, builder_for_with, follows_calendar, BarBuilder, BarSpec, BarStamp,
    BuilderParams, SourceRecord,
};
use crate::portfolio::kernel::KernelBar;
use crate::runtime::ids::{AggId, InstrumentId, StrategyId};
use crate::runtime::memory::HeapSize;
use crate::runtime::router::{AggSubscriptions, AggSubscriptionsBuilder};

/// Everything that changes an aggregator's output.
///
/// Built through [`AggKey::new`], which folds parameters that cannot affect
/// the output of the chosen unit into one value, so two subscriptions share an
/// aggregator exactly when their output streams are identical:
///
/// * `brick_bits` is the *resolved* Renko brick (`brick_size` when positive,
///   else the spec's step) for Renko and `0` for every other unit. `-0.0`,
///   `0.0`, negatives and NaN all resolve to the step, so none of them can
///   split an aggregator; an infinite brick is refused.
/// * `tz_offset_ns` only moves time-window boundaries, so it is `0` for
///   count, volume, value, Renko and flow units.
/// * `stamp` only changes how a bar's timestamp picks its window, so it is
///   `Open` for non-time-window units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AggKey {
    instrument: InstrumentId,
    spec: BarSpec,
    brick_bits: u64,
    tz_offset_ns: i64,
    stamp: BarStamp,
}

impl AggKey {
    pub fn new(
        instrument: InstrumentId,
        spec: BarSpec,
        brick_size: f64,
        tz_offset_ns: i64,
        stamp: BarStamp,
    ) -> Result<Self> {
        let brick_bits = if spec.unit == crate::data::AggregationUnit::Renko {
            BuilderParams { brick_size }
                .resolved_brick(spec)
                .map_err(|e| RaptorError::invalid_parameter(e.to_string()))?
                .to_bits()
        } else {
            0
        };
        let time_window = spec.unit.is_time_window();
        Ok(Self {
            instrument,
            spec,
            brick_bits,
            tz_offset_ns: if time_window { tz_offset_ns } else { 0 },
            stamp: if time_window { stamp } else { BarStamp::Open },
        })
    }

    pub fn instrument(&self) -> InstrumentId {
        self.instrument
    }

    pub fn spec(&self) -> BarSpec {
        self.spec
    }

    pub fn brick_bits(&self) -> u64 {
        self.brick_bits
    }

    pub fn tz_offset_ns(&self) -> i64 {
        self.tz_offset_ns
    }

    pub fn stamp(&self) -> BarStamp {
        self.stamp
    }

    fn params(&self) -> BuilderParams {
        BuilderParams { brick_size: f64::from_bits(self.brick_bits) }
    }
}

/// Registration phase: dedups keys and records each strategy's subscriptions.
pub struct AggregatorHubBuilder {
    instruments: usize,
    by_key: HashMap<AggKey, AggId>,
    keys: Vec<AggKey>,
    builders: Vec<Box<dyn BarBuilder + Send + Sync>>,
    /// Per strategy, its subscriptions in order; the position is the local
    /// stream id.
    strategies: Vec<Vec<AggId>>,
    /// Per instrument, the builder of its primary bars from trades.
    primary: Vec<Option<Box<dyn BarBuilder + Send + Sync>>>,
}

impl AggregatorHubBuilder {
    pub fn new(instruments: usize) -> Self {
        Self {
            instruments,
            by_key: HashMap::new(),
            keys: Vec::new(),
            builders: Vec::new(),
            strategies: Vec::new(),
            primary: (0..instruments).map(|_| None).collect(),
        }
    }

    /// Build `instrument`'s primary bars from trade prints: `spec` bars cut
    /// at `tz_offset_ns`'s midnight for daily and longer units. Replaces an
    /// earlier setting.
    pub fn set_primary(
        &mut self,
        instrument: InstrumentId,
        spec: BarSpec,
        tz_offset_ns: i64,
    ) -> Result<()> {
        let slot = self.primary.get_mut(instrument.index()).ok_or_else(|| {
            RaptorError::invalid_parameter(format!(
                "instrument id {} is outside the {} registered instruments",
                instrument.raw(),
                self.instruments
            ))
        })?;
        let builder = builder_for_with(spec, tz_offset_ns, BuilderParams { brick_size: 0.0 })
            .map_err(|e| RaptorError::invalid_parameter(e.to_string()))?;
        *slot = Some(builder);
        Ok(())
    }

    /// Subscribe `strategy` to `key`; returns the shared aggregator and the
    /// strategy's local stream id for this subscription.
    ///
    /// `AggId`s follow first registration. Subscribing twice to one key gives
    /// two local streams on the same aggregator.
    pub fn register(&mut self, strategy: StrategyId, key: AggKey) -> Result<(AggId, u16)> {
        self.register_with_session(strategy, key, None)
    }

    /// [`AggregatorHubBuilder::register`] for an instrument that has a
    /// trading calendar. Daily and longer aggregations then cut at the
    /// calendar's own midnight and skip non-trading days; the key's
    /// `tz_offset_ns` is replaced accordingly so two subscriptions still
    /// share an aggregator exactly when their output is identical.
    pub fn register_with_session(
        &mut self,
        strategy: StrategyId,
        key: AggKey,
        session: Option<&SessionSchedule>,
    ) -> Result<(AggId, u16)> {
        let session = session.filter(|_| follows_calendar(key.spec.unit));
        let key = match session {
            Some(schedule) => AggKey { tz_offset_ns: schedule.tz_offset_ns(), ..key },
            None => key,
        };
        if key.instrument.index() >= self.instruments {
            return Err(RaptorError::invalid_parameter(format!(
                "instrument id {} is outside the {} registered instruments",
                key.instrument.raw(),
                self.instruments
            )));
        }
        let agg = match self.by_key.get(&key) {
            Some(&agg) => agg,
            None => {
                let builder =
                    builder_for_session(key.spec, key.tz_offset_ns, key.params(), session)
                        .map_err(|e| RaptorError::invalid_parameter(e.to_string()))?;
                let raw = u32::try_from(self.keys.len())
                    .map_err(|_| RaptorError::invalid_parameter("too many aggregations"))?;
                let agg = AggId::new(raw);
                self.by_key.insert(key, agg);
                self.keys.push(key);
                self.builders.push(builder);
                agg
            }
        };
        if self.strategies.len() <= strategy.index() {
            self.strategies.resize_with(strategy.index() + 1, Vec::new);
        }
        let subs = &mut self.strategies[strategy.index()];
        let local = u16::try_from(subs.len()).map_err(|_| {
            RaptorError::invalid_parameter("a strategy may hold at most 65536 subscriptions")
        })?;
        subs.push(agg);
        Ok((agg, local))
    }

    /// Freeze the registrations into a hub. The dedup map is dropped.
    pub fn build(self) -> Result<AggregatorHub> {
        let aggs = self.keys.len();

        // Aggregations per instrument, ascending AggId.
        let mut inst_offsets = vec![0u32; self.instruments + 1];
        for key in &self.keys {
            inst_offsets[key.instrument.index() + 1] += 1;
        }
        for i in 0..self.instruments {
            inst_offsets[i + 1] += inst_offsets[i];
        }
        let mut cursor = inst_offsets.clone();
        let mut inst_aggs = vec![AggId::new(0); aggs];
        for (i, key) in self.keys.iter().enumerate() {
            let slot = &mut cursor[key.instrument.index()];
            inst_aggs[*slot as usize] = AggId::new(i as u32);
            *slot += 1;
        }

        // Each strategy's subscriptions, flat.
        let mut strat_offsets = Vec::with_capacity(self.strategies.len() + 1);
        let mut strat_subs = Vec::new();
        let mut agg_subs = AggSubscriptionsBuilder::new(aggs);
        strat_offsets.push(0u32);
        for (s, subs) in self.strategies.iter().enumerate() {
            for (local, &agg) in subs.iter().enumerate() {
                agg_subs.push(agg, StrategyId::new(s as u32), local as u16)?;
            }
            strat_subs.extend_from_slice(subs);
            strat_offsets.push(u32::try_from(strat_subs.len()).map_err(|_| {
                RaptorError::invalid_parameter("too many subscriptions (u32 offsets)")
            })?);
        }

        Ok(AggregatorHub {
            by_key: self.by_key,
            keys: self.keys,
            builders: self.builders,
            inst_offsets,
            inst_aggs,
            strat_offsets,
            strat_subs,
            agg_subs: agg_subs.build()?,
            spans: vec![(0, 0); aggs],
            emissions: Vec::new(),
            current: None,
            push_calls: 0,
            primary: self.primary,
        })
    }
}

/// The shared aggregators of a run.
pub struct AggregatorHub {
    /// Kept after the build so an instrument added to a running node can share an aggregator
    /// with a key that already exists.
    by_key: HashMap<AggKey, AggId>,
    keys: Vec<AggKey>,
    builders: Vec<Box<dyn BarBuilder + Send + Sync>>,
    /// Aggregations per instrument, ascending `AggId`.
    inst_offsets: Vec<u32>,
    inst_aggs: Vec<AggId>,
    /// Each strategy's aggregation per local stream.
    strat_offsets: Vec<u32>,
    strat_subs: Vec<AggId>,
    agg_subs: AggSubscriptions,
    /// Per `AggId`: range of the last event's emissions. Only entries of the
    /// current instrument are meaningful.
    spans: Vec<(u32, u32)>,
    /// Reused across events.
    emissions: Vec<(AggId, OhlcvBar)>,
    current: Option<InstrumentId>,
    push_calls: u64,
    primary: Vec<Option<Box<dyn BarBuilder + Send + Sync>>>,
}

/// What a retired instrument's aggregator is replaced with, so its state is freed.
#[derive(Debug)]
struct Retired;

impl BarBuilder for Retired {
    fn push(&mut self, _rec: &SourceRecord) -> Option<OhlcvBar> {
        None
    }

    fn flush(&mut self) -> Option<OhlcvBar> {
        None
    }
}

impl AggregatorHub {
    /// Make room for one more instrument (the next id), with no aggregations. Its primary bars
    /// are cut as `primary` says, if given.
    pub fn add_instrument(&mut self, primary: Option<(BarSpec, i64)>) -> Result<()> {
        let end = *self.inst_offsets.last().unwrap_or(&0);
        if self.inst_offsets.is_empty() {
            self.inst_offsets.push(0);
        }
        self.inst_offsets.push(end);
        let builder = match primary {
            Some((spec, tz)) => Some(
                builder_for_with(spec, tz, BuilderParams { brick_size: 0.0 })
                    .map_err(|e| RaptorError::invalid_parameter(e.to_string()))?,
            ),
            None => None,
        };
        self.primary.push(builder);
        Ok(())
    }

    /// Subscribe `strategy` to `key` after the build. Only the newest instrument may gain
    /// aggregations (the index keeps each instrument's aggregations in `AggId` order, so a new
    /// aggregator must belong to the last instrument); the stream is appended to the strategy's
    /// own list. Returns what [`AggregatorHubBuilder::register_with_session`] returns.
    pub fn subscribe(
        &mut self,
        strategy: StrategyId,
        key: AggKey,
        session: Option<&SessionSchedule>,
    ) -> Result<(AggId, u16)> {
        let session = session.filter(|_| follows_calendar(key.spec.unit));
        let key = match session {
            Some(schedule) => AggKey { tz_offset_ns: schedule.tz_offset_ns(), ..key },
            None => key,
        };
        let instruments = self.inst_offsets.len().saturating_sub(1);
        if key.instrument.index() + 1 != instruments {
            return Err(RaptorError::invalid_parameter(
                "a running node can subscribe only its newest instrument to a new aggregation",
            ));
        }
        let agg = match self.by_key.get(&key) {
            Some(&agg) => agg,
            None => {
                let builder =
                    builder_for_session(key.spec, key.tz_offset_ns, key.params(), session)
                        .map_err(|e| RaptorError::invalid_parameter(e.to_string()))?;
                let agg = AggId::new(
                    u32::try_from(self.keys.len())
                        .map_err(|_| RaptorError::invalid_parameter("too many aggregations"))?,
                );
                self.by_key.insert(key, agg);
                self.keys.push(key);
                self.builders.push(builder);
                self.spans.push((0, 0));
                // The newest instrument's aggregations are the tail of `inst_aggs`.
                self.inst_aggs.push(agg);
                *self.inst_offsets.last_mut().expect("an instrument exists") += 1;
                agg
            }
        };
        let s = strategy.index();
        if self.strat_offsets.is_empty() {
            self.strat_offsets.push(0);
        }
        while self.strat_offsets.len() <= s + 1 {
            let end = *self.strat_offsets.last().expect("non-empty");
            self.strat_offsets.push(end);
        }
        let at = self.strat_offsets[s + 1] as usize;
        let local = u16::try_from(at - self.strat_offsets[s] as usize).map_err(|_| {
            RaptorError::invalid_parameter("a strategy may hold at most 65536 subscriptions")
        })?;
        self.strat_subs.insert(at, agg);
        for o in &mut self.strat_offsets[s + 1..] {
            *o += 1;
        }
        let mut rows: Vec<(AggId, StrategyId, u16)> = Vec::with_capacity(self.strat_subs.len());
        let mut agg_subs = AggSubscriptionsBuilder::new(self.keys.len());
        for st in 0..self.strat_offsets.len() - 1 {
            let (lo, hi) = (self.strat_offsets[st] as usize, self.strat_offsets[st + 1] as usize);
            for (l, &a) in self.strat_subs[lo..hi].iter().enumerate() {
                rows.push((a, StrategyId::new(st as u32), l as u16));
            }
        }
        for (a, st, l) in rows {
            agg_subs.push(a, st, l)?;
        }
        self.agg_subs = agg_subs.build()?;
        Ok((agg, local))
    }

    /// Stop feeding `instrument`'s aggregators and free their state. The ids stay (they are
    /// never reused), so every strategy's stream numbering is unchanged.
    pub fn retire_instrument(&mut self, instrument: InstrumentId) -> Result<()> {
        let i = instrument.index();
        let (Some(&lo), Some(&hi)) = (self.inst_offsets.get(i), self.inst_offsets.get(i + 1))
        else {
            return Ok(());
        };
        let freed = hi - lo;
        for k in lo..hi {
            let a = self.inst_aggs[k as usize].index();
            self.builders[a] = Box::new(Retired);
        }
        self.inst_aggs.drain(lo as usize..hi as usize);
        for o in &mut self.inst_offsets[i + 1..] {
            *o -= freed;
        }
        if let Some(slot) = self.primary.get_mut(i) {
            *slot = None;
        }
        Ok(())
    }

    /// Bring a retired instrument's aggregators back with fresh builders (the ids and every
    /// strategy's stream numbering are the ones they had), and its primary bars if `primary` says.
    pub fn revive_instrument(
        &mut self,
        instrument: InstrumentId,
        primary: Option<(BarSpec, i64)>,
        session: Option<&SessionSchedule>,
    ) -> Result<()> {
        let i = instrument.index();
        let Some(&lo) = self.inst_offsets.get(i) else {
            return Err(RaptorError::invalid_parameter("unknown instrument"));
        };
        if self.inst_offsets[i + 1] != lo {
            return Err(RaptorError::invalid_parameter("the instrument's aggregators are live"));
        }
        let mine: Vec<usize> =
            (0..self.keys.len()).filter(|&a| self.keys[a].instrument == instrument).collect();
        for &a in &mine {
            let key = self.keys[a];
            let session = session.filter(|_| follows_calendar(key.spec.unit));
            self.builders[a] =
                builder_for_session(key.spec, key.tz_offset_ns, key.params(), session)
                    .map_err(|e| RaptorError::invalid_parameter(e.to_string()))?;
            self.spans[a] = (0, 0);
        }
        let ids: Vec<AggId> = mine.iter().map(|&a| AggId::new(a as u32)).collect();
        self.inst_aggs.splice(lo as usize..lo as usize, ids.iter().copied());
        for o in &mut self.inst_offsets[i + 1..] {
            *o += ids.len() as u32;
        }
        if let (Some((spec, tz)), Some(slot)) = (primary, self.primary.get_mut(i)) {
            *slot = Some(
                builder_for_with(spec, tz, BuilderParams { brick_size: 0.0 })
                    .map_err(|e| RaptorError::invalid_parameter(e.to_string()))?,
            );
        }
        Ok(())
    }

    /// Feed one primary bar of `instrument`.
    pub fn on_bar(&mut self, instrument: InstrumentId, bar: &KernelBar) -> &[(AggId, OhlcvBar)] {
        let rec = SourceRecord {
            timestamp: bar.timestamp,
            open: bar.open,
            high: bar.high,
            low: bar.low,
            close: bar.close,
            volume: bar.volume,
            signed_volume: 0.0,
        };
        self.on_record(instrument, &rec)
    }

    /// Feed one primary bar given as a record; its timestamp carries each
    /// aggregator's own [`BarStamp`] meaning.
    ///
    /// Pushes the record once into every aggregator of `instrument` in
    /// `AggId` order, draining held-back Renko bricks after each completed
    /// bar. Returns the emissions in that order. The buffer is reused, so
    /// steady-state calls do not allocate.
    pub fn on_record(
        &mut self,
        instrument: InstrumentId,
        rec: &SourceRecord,
    ) -> &[(AggId, OhlcvBar)] {
        self.push_record(instrument, rec, true)
    }

    /// Feed one trade print of `instrument` to its aggregators, as
    /// `BarAggregator.push_trade(ts, price, size)` does: the print is an
    /// instant, so no [`BarStamp`] applies to it. Emissions as
    /// [`on_record`](Self::on_record).
    pub fn on_trade(
        &mut self,
        instrument: InstrumentId,
        timestamp: i64,
        price: f64,
        size: f64,
    ) -> &[(AggId, OhlcvBar)] {
        self.push_record(instrument, &SourceRecord::trade(timestamp, price, size), false)
    }

    /// Feed one print to `instrument`'s primary builder (see
    /// [`AggregatorHubBuilder::set_primary`]); the bar it completes, if any.
    /// A Renko brick held back behind the first is not drained, as in a tick
    /// run.
    pub fn primary_trade(
        &mut self,
        instrument: InstrumentId,
        timestamp: i64,
        price: f64,
        size: f64,
    ) -> Option<OhlcvBar> {
        let builder = self.primary.get_mut(instrument.index())?.as_mut()?;
        builder.push(&SourceRecord::trade(timestamp, price, size))
    }

    /// An event that feeds no aggregator (a quote, a book): nothing is
    /// emitted for `instrument` until the next feed.
    pub fn skip(&mut self, instrument: InstrumentId) {
        self.emissions.clear();
        self.current = Some(instrument);
    }

    fn push_record(
        &mut self,
        instrument: InstrumentId,
        rec: &SourceRecord,
        stamped: bool,
    ) -> &[(AggId, OhlcvBar)] {
        self.emissions.clear();
        self.current = Some(instrument);
        let i = instrument.index();
        if let (Some(&lo), Some(&hi)) = (self.inst_offsets.get(i), self.inst_offsets.get(i + 1)) {
            for &agg in &self.inst_aggs[lo as usize..hi as usize] {
                let a = agg.index();
                let key = &self.keys[a];
                let mut r = *rec;
                if stamped {
                    r.timestamp = key.stamp.bucket_ts(key.spec.unit, rec.timestamp);
                }
                let start = self.emissions.len();
                let builder = &mut self.builders[a];
                self.push_calls += 1;
                if let Some(bar) = builder.push(&r) {
                    self.emissions.push((agg, bar));
                    // Renko completes several bricks from one record.
                    while let Some(bar) = builder.next_pending() {
                        self.emissions.push((agg, bar));
                    }
                }
                self.spans[a] = (start as u32, self.emissions.len() as u32);
            }
        }
        &self.emissions
    }

    /// The last event's emissions, in emission order.
    pub fn emissions(&self) -> &[(AggId, OhlcvBar)] {
        &self.emissions
    }

    /// What `strategy` receives for the last event: its own subscriptions on
    /// that instrument, in subscription order, each with its aggregator's
    /// emissions, as `(local_stream, bar)`. A subscription held twice
    /// delivers twice.
    pub fn deliveries(&self, strategy: StrategyId) -> Deliveries<'_> {
        let i = strategy.index();
        // Most events complete nothing: skip the scan of the strategy's list.
        let subs = match (self.strat_offsets.get(i), self.strat_offsets.get(i + 1)) {
            (Some(&lo), Some(&hi)) if !self.emissions.is_empty() => {
                &self.strat_subs[lo as usize..hi as usize]
            }
            _ => &[],
        };
        Deliveries { hub: self, subs, stream: 0, next: 0 }
    }

    /// `(strategy, local stream)` subscribed to an aggregation.
    pub fn subscribers(&self, agg: AggId) -> &[(StrategyId, u16)] {
        self.agg_subs.subscribers(agg)
    }

    pub fn key(&self, agg: AggId) -> Option<&AggKey> {
        self.keys.get(agg.index())
    }

    /// Distinct aggregators.
    pub fn aggregators(&self) -> usize {
        self.keys.len()
    }

    /// Subscriptions across all strategies.
    pub fn subscriptions(&self) -> usize {
        self.strat_subs.len()
    }

    /// Heap held by per-strategy subscription entries: each strategy's
    /// list of aggregations and the aggregation-to-subscriber index.
    pub fn subscription_heap_bytes(&self) -> usize {
        use std::mem::size_of;
        self.strat_offsets.capacity() * size_of::<u32>()
            + self.strat_subs.capacity() * size_of::<AggId>()
            + self.agg_subs.heap_bytes()
    }

    /// Builder `push` calls so far: one per aggregator per event.
    pub fn push_calls(&self) -> u64 {
        self.push_calls
    }
}

impl HeapSize for AggregatorHub {
    /// Index and buffer bytes. Each builder's own state sits behind a trait
    /// object and is not counted.
    fn heap_bytes(&self) -> usize {
        use std::mem::size_of;
        self.keys.capacity() * size_of::<AggKey>()
            + self.by_key.capacity() * (size_of::<AggKey>() + size_of::<AggId>())
            + self.builders.capacity() * size_of::<Box<dyn BarBuilder + Send + Sync>>()
            + self.inst_offsets.capacity() * size_of::<u32>()
            + self.inst_aggs.capacity() * size_of::<AggId>()
            + self.strat_offsets.capacity() * size_of::<u32>()
            + self.strat_subs.capacity() * size_of::<AggId>()
            + self.agg_subs.heap_bytes()
            + self.spans.capacity() * size_of::<(u32, u32)>()
            + self.primary.capacity() * size_of::<Option<Box<dyn BarBuilder + Send + Sync>>>()
            + self.emissions.capacity() * size_of::<(AggId, OhlcvBar)>()
    }
}

/// One strategy's deliveries for the last event; see
/// [`AggregatorHub::deliveries`].
pub struct Deliveries<'a> {
    hub: &'a AggregatorHub,
    subs: &'a [AggId],
    /// Position in `subs`, which is the local stream id.
    stream: usize,
    /// Next emission within the current subscription's span.
    next: u32,
}

impl<'a> Iterator for Deliveries<'a> {
    type Item = (u16, &'a OhlcvBar);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(&agg) = self.subs.get(self.stream) {
            let a = agg.index();
            if Some(self.hub.keys[a].instrument) == self.hub.current {
                let (lo, hi) = self.hub.spans[a];
                if lo + self.next < hi {
                    let bar = &self.hub.emissions[(lo + self.next) as usize].1;
                    self.next += 1;
                    return Some((self.stream as u16, bar));
                }
            }
            self.stream += 1;
            self.next = 0;
        }
        None
    }
}
