//! Fee calculation models.

use std::collections::{BTreeSet, HashMap, VecDeque};
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::core::types::{Direction, Price};
use crate::execution::costs::{FeeBreakdown, FeeRates, FeeSchedule, FeeSegment};
use crate::execution::indian_costs::Segment;
use crate::instruments::{InstrumentKind, InstrumentSpec};

/// What a [`fee_gap_fault`] message starts with.
pub const FEE_GAP_PREFIX: &str = "fee_schedule_gap:";

/// A fill (or settlement) priced on a date the fee schedule has no rate for. Like a money
/// fault it unwinds out of the kernel as a panic whose message starts `fee_schedule_gap:`,
/// which the Python layer (`raptorbt._guard`) turns into the structured error of that code:
/// the run is refused rather than charged a rate that was not in force.
#[cold]
#[inline(never)]
pub fn fee_gap_fault(detail: impl std::fmt::Display) -> ! {
    panic!("{FEE_GAP_PREFIX} {detail}")
}

/// No trading date has paid the per-day fee yet.
const NO_DAY: i64 = i64::MIN;

const DAY_NS: i64 = 86_400_000_000_000;

/// The schedule a run is priced on, and optionally the one row family that
/// prices every instrument; the kernels each build their [`DatedFees`] from
/// it.
///
/// With `fixed` set, every instrument is priced on that venue and segment
/// (an explicit override). Without it each instrument picks its own row from
/// its venue and kind ([`fee_row_for`]): a future is never charged the
/// option row.
#[derive(Debug, Clone)]
pub struct FeeBinding {
    pub schedule: Arc<FeeSchedule>,
    pub fixed: Option<(String, FeeSegment)>,
}

impl FeeBinding {
    /// The unverified rows of the fixed venue and segment whose dates overlap
    /// `first_ts..=last_ts` (UTC ns), for runners that price one row and keep
    /// no per-fill record of which rows they used.
    pub fn unverified_between(&self, first_ts: i64, last_ts: i64) -> Vec<String> {
        let Some((venue, segment)) = &self.fixed else { return Vec::new() };
        let mut out: Vec<String> = self
            .schedule
            .rows
            .iter()
            .filter(|r| {
                &r.venue == venue
                    && r.segment == *segment
                    && !r.settlement
                    && r.meta.status == crate::execution::costs::RowStatus::Unverified
                    && r.effective_from <= last_ts
                    && r.meta.effective_until.is_none_or(|u| u > first_ts)
            })
            .map(|r| r.label())
            .collect();
        out.sort();
        out.dedup();
        out
    }
}

/// The venue and fee segment an instrument is priced on: its venue, and by
/// kind a future (a contract), an option, or cash equity. Cash is priced as
/// delivery unless `intraday`, which a run sets when its positions are
/// squared off the same day. Errors name what the instrument lacks.
pub fn fee_row_for(
    spec: &InstrumentSpec,
    intraday: bool,
    schedule: &FeeSchedule,
) -> Result<(String, FeeSegment, Option<String>), String> {
    let venue = spec.venue.clone().ok_or_else(|| {
        format!(
            "instrument {:?} has no venue, so no fee row can be chosen for it: give the \
             InstrumentSpec a venue (\"NSE\", \"NFO\", \"MCX\", ...) or set fee_segment",
            spec.symbol
        )
    })?;
    let venue = venue.to_ascii_uppercase();
    let segment = match &spec.kind {
        InstrumentKind::Option { .. } => FeeSegment::Options,
        InstrumentKind::Contract { .. } => FeeSegment::Futures,
        InstrumentKind::Cash | InstrumentKind::CurrencyPair if intraday => FeeSegment::Intraday,
        InstrumentKind::Cash | InstrumentKind::CurrencyPair => FeeSegment::Delivery,
        InstrumentKind::Index => {
            return Err(format!("instrument {:?} is an index and is not traded", spec.symbol))
        }
    };
    // MCX: the agricultural row (CTT excluded, SEBI Re 1 per crore) applies to the contracts the
    // session table calls agricultural; a name in no class of that table is refused, never
    // charged the non-agricultural row by default.
    if venue == "MCX"
        && schedule.rows.iter().any(|r| {
            r.venue == venue
                && r.segment == segment
                && r.underlying.as_deref() == Some(crate::execution::costs::schedule::MCX_AGRI)
        })
    {
        use crate::core::session_schedule::{mcx_class, McxClass};
        let mut names: Vec<&str> = spec.underlying().into_iter().collect();
        names.push(spec.symbol.as_str());
        return match names.iter().find_map(|n| mcx_class(n)) {
            Some(McxClass::SelectAgri | McxClass::OtherAgri) => {
                Ok((venue, segment, Some(crate::execution::costs::schedule::MCX_AGRI.to_string())))
            }
            Some(McxClass::Standard) => Ok((venue, segment, None)),
            None => Err(format!(
                "instrument {:?}: no MCX fee class is known for {:?}, so its charges cannot be \
                 chosen (agricultural contracts pay no commodities transaction tax and SEBI Re 1 \
                 per crore; bullion, base metals and energy pay both at the general row). Name an \
                 underlying the session table lists, or pass a fee_schedule with a row for it",
                spec.symbol,
                names.first().copied().unwrap_or_default(),
            )),
        };
    }
    // Rows keyed by scrip group (BSE cash) need the instrument's group, never an assumed one.
    let groups: Vec<&str> = schedule
        .rows
        .iter()
        .filter(|r| r.venue == venue && r.segment == segment)
        .filter_map(|r| r.underlying.as_deref()?.strip_prefix("group:"))
        .collect();
    if groups.is_empty() {
        return Ok((venue, segment, None));
    }
    let mut groups = groups;
    groups.sort_unstable();
    groups.dedup();
    match spec.scrip_group.as_deref().map(str::to_ascii_uppercase) {
        Some(g) if groups.contains(&g.as_str()) => Ok((venue, segment, Some(format!("group:{g}")))),
        Some(g) => Err(format!(
            "instrument {:?}: {venue} charges are not known for scrip group {g:?}; the groups priced are {}",
            spec.symbol,
            groups.join(", ")
        )),
        None => Err(format!(
            "instrument {:?}: {venue} charges depend on the scrip group; set the InstrumentSpec \
             scrip_group to one of {}",
            spec.symbol,
            groups.join(", ")
        )),
    }
}

/// A fee schedule bound to the venue, segment and underlying of one
/// instrument, priced by the row in force on the day of each fill (a fill on
/// day D uses the row whose `effective_from` is the latest at or before D),
/// with the contract's settlement row and the per-instrument-day fee.
///
/// The kernel moves it to each step's time with [`FeeModel::set_time`]; every
/// fill of that step is then priced on that row. The per-day fee is owed by
/// the first fill of a trading date that carries it, and the date is paid
/// once a booked fill carried it ([`FeeModel::note_booked`]).
#[derive(Debug)]
pub struct DatedFees {
    pub schedule: Arc<FeeSchedule>,
    pub venue: String,
    pub segment: FeeSegment,
    /// The instrument's underlying symbol, for rows that name one.
    pub underlying: Option<String>,
    /// UTC offset (ns) that puts a timestamp on the venue's trading date.
    pub tz_offset_ns: i64,
    /// The row family is chosen from the instrument, not fixed by the run.
    per_instrument: bool,
    /// Why no row could be chosen for the instrument, when none could.
    unresolved: Option<String>,
    /// The row key of the instrument's scrip group (`group:A`), for the delivery of
    /// the stock a BSE stock derivative settles in.
    scrip_group: Option<String>,
    current: FeeRates,
    /// Trading date of the step in progress.
    day: i64,
    /// Trading date that last paid the per-day fee.
    paid_day: AtomicI64,
    /// The order whose fills are being booked (id + 1; 0 for none), so a
    /// brokerage a card charges per order is charged once over its fills.
    order_key: AtomicU64,
    /// What the fills of each order booked so far, by order id.
    order_turnover: Mutex<HashMap<u64, f64>>,
    /// The value of the charge computed last, which booking adds to the order.
    last_value: AtomicU64,
    /// For the most recent booked fills of an order: what its earlier fills
    /// had traded when this one was booked, kept (up to [`RECENT_SLICES`]) so a
    /// venue simulation pricing the same fill afterwards charges it the same.
    slices: Mutex<VecDeque<(u64, f64)>>,
    /// Labels of the rows without an official source this model priced fills on.
    unverified: Mutex<BTreeSet<String>>,
    /// Why the trading date in progress has no rate, when it has none: pricing a fill on it
    /// stops with [`fee_gap_fault`]. Dates without a fill never charge, so they never fail.
    gap: Option<String>,
    /// Cash equity bound to the delivery row whose same-day round trips are priced on the
    /// intraday row ([`FeeModel::same_day_charges`]).
    classify_by_day: bool,
}

/// How many booked fills' order context is kept for [`FeeModel::take_slice_before`].
const RECENT_SLICES: usize = 4096;

/// What the booked legs of a same-day round trip paid in brokerage (see
/// [`FeeModel::same_day_charges`]).
#[derive(Debug, Clone, Copy)]
pub struct SameDayBrokerage {
    /// The entry's brokerage when its itemised charges are known.
    pub entry: Option<f64>,
    /// What the entry was booked at in all, from which the brokerage is read when `entry` is
    /// `None` (a slice of a position that was scaled into or partly closed).
    pub entry_total: f64,
    /// The exit's brokerage, as booked.
    pub exit: f64,
}

impl DatedFees {
    /// The brokerage an entry booked at `booked_total` on the delivery row paid: the total less
    /// what the row charges on `value` with no brokerage, over what one rupee of brokerage adds
    /// (GST on it). Read back, not recomputed, so a position built of several fills or sliced
    /// into several exits keeps what each order was charged.
    fn booked_entry_brokerage(&self, day: i64, is_buy: bool, value: f64, booked_total: f64) -> f64 {
        // The entry was booked on this same row (same venue, segment and day), so a lookup that
        // fails now is a schedule changed under the run: stop, as every other fee lookup does,
        // rather than read the entry's brokerage as nothing and understate the round trip.
        let row = match self.schedule.row_for(
            &self.venue,
            self.segment,
            self.underlying.as_deref(),
            day,
            false,
        ) {
            Ok(row) => row,
            Err(why) => fee_gap_fault(why),
        };
        let without = row.rates.charge_keeping_brokerage(is_buy, value, 0.0).total();
        let slope = row.rates.charge_keeping_brokerage(is_buy, value, 1.0).total() - without;
        if slope <= 0.0 {
            return 0.0;
        }
        ((booked_total - without) / slope).max(0.0)
    }
}

impl Clone for DatedFees {
    fn clone(&self) -> Self {
        Self {
            schedule: Arc::clone(&self.schedule),
            venue: self.venue.clone(),
            segment: self.segment,
            underlying: self.underlying.clone(),
            tz_offset_ns: self.tz_offset_ns,
            per_instrument: self.per_instrument,
            unresolved: self.unresolved.clone(),
            scrip_group: self.scrip_group.clone(),
            current: self.current,
            day: self.day,
            paid_day: AtomicI64::new(self.paid_day.load(Ordering::Relaxed)),
            order_key: AtomicU64::new(self.order_key.load(Ordering::Relaxed)),
            order_turnover: Mutex::new(self.turnover_map().clone()),
            last_value: AtomicU64::new(self.last_value.load(Ordering::Relaxed)),
            slices: Mutex::new(self.slice_log().clone()),
            unverified: Mutex::new(self.unverified_set().clone()),
            gap: self.gap.clone(),
            classify_by_day: self.classify_by_day,
        }
    }
}

impl DatedFees {
    /// Bind a schedule row family; priced on the latest row until the first
    /// step says otherwise.
    pub fn new(
        schedule: Arc<FeeSchedule>,
        venue: String,
        segment: FeeSegment,
        tz_offset_ns: i64,
    ) -> crate::core::error::Result<Self> {
        let current = schedule.row_for(&venue, segment, None, i64::MAX, false)?.rates;
        Ok(Self {
            schedule,
            venue,
            segment,
            underlying: None,
            tz_offset_ns,
            per_instrument: false,
            unresolved: None,
            scrip_group: None,
            current,
            day: NO_DAY,
            paid_day: AtomicI64::new(NO_DAY),
            order_key: AtomicU64::new(0),
            order_turnover: Mutex::new(HashMap::new()),
            last_value: AtomicU64::new(0),
            slices: Mutex::new(VecDeque::new()),
            unverified: Mutex::new(BTreeSet::new()),
            gap: None,
            classify_by_day: false,
        })
    }

    /// A schedule whose row is chosen from each instrument it prices (see
    /// [`DatedFees::bind_instrument`]). Charging before an instrument is
    /// bound is a bug, not a free fill.
    pub fn per_instrument(schedule: Arc<FeeSchedule>, tz_offset_ns: i64) -> Self {
        Self {
            schedule,
            venue: String::new(),
            segment: FeeSegment::Futures,
            underlying: None,
            tz_offset_ns,
            per_instrument: true,
            unresolved: Some("no instrument has been attached to the kernel".to_string()),
            scrip_group: None,
            current: FeeRates::ZERO,
            day: NO_DAY,
            paid_day: AtomicI64::new(NO_DAY),
            order_key: AtomicU64::new(0),
            order_turnover: Mutex::new(HashMap::new()),
            last_value: AtomicU64::new(0),
            slices: Mutex::new(VecDeque::new()),
            unverified: Mutex::new(BTreeSet::new()),
            gap: None,
            classify_by_day: false,
        }
    }

    /// Choose the row family from the instrument (when the run did not fix
    /// one), and the underlying for rows that name one. The error says why
    /// no row fits; the model then refuses to charge.
    pub fn bind_instrument(&mut self, spec: &InstrumentSpec, intraday: bool) -> Result<(), String> {
        self.underlying = spec.underlying().map(str::to_owned);
        self.scrip_group =
            spec.scrip_group.as_ref().map(|g| format!("group:{}", g.to_ascii_uppercase()));
        self.day = NO_DAY;
        self.gap = None;
        self.classify_by_day = false;
        // A fill's row is the one in force on the venue's own trading date.
        if let Some(session) = &spec.session {
            self.tz_offset_ns = session.tz_offset_ns();
        }
        if !self.per_instrument {
            if let Some(r) = self.rates_at(i64::MAX) {
                self.current = r;
            }
            return Ok(());
        }
        let resolved =
            fee_row_for(spec, intraday, &self.schedule).and_then(|(venue, segment, group)| {
                if group.is_some() {
                    self.underlying = group;
                }
                self.schedule
                    .row_for(&venue, segment, self.underlying.as_deref(), i64::MAX, false)
                    .map(|row| (venue, segment, row.rates))
                    .map_err(|e| format!("instrument {:?}: {e}", spec.symbol))
            });
        match resolved {
            Ok((venue, segment, rates)) => {
                // Cash equity priced as delivery is repriced on the intraday row when it is
                // opened and closed on one trading date, if the schedule has that row.
                self.classify_by_day = segment == FeeSegment::Delivery
                    && matches!(spec.kind, InstrumentKind::Cash | InstrumentKind::CurrencyPair)
                    && matches!(
                        self.schedule.row_for_opt(
                            &venue,
                            FeeSegment::Intraday,
                            self.underlying.as_deref(),
                            i64::MAX,
                            false
                        ),
                        Ok(Some(_))
                    );
                self.venue = venue;
                self.segment = segment;
                self.current = rates;
                self.unresolved = None;
                Ok(())
            }
            Err(why) => {
                self.unresolved = Some(why.clone());
                Err(why)
            }
        }
    }

    /// The charging rates; an unbound per-instrument model is a bug in the
    /// caller (the bindings check each instrument when it is registered), so
    /// it stops with the reason instead of charging nothing.
    fn rates(&self) -> &FeeRates {
        if let Some(why) = &self.unresolved {
            panic!("fee schedule {:?} cannot price this instrument: {why}", self.schedule.name);
        }
        if let Some(why) = &self.gap {
            fee_gap_fault(why);
        }
        &self.current
    }

    /// The row in force at `ts` (UTC ns), for the bound underlying.
    fn rates_at(&self, ts: i64) -> Option<FeeRates> {
        self.schedule
            .row_for(&self.venue, self.segment, self.underlying.as_deref(), ts, false)
            .ok()
            .map(|r| r.rates)
    }

    /// [`DatedFees::rates_at`] for a trading date being priced: a row without
    /// an official source is remembered as used.
    fn rates_used_at(&self, ts: i64) -> Result<FeeRates, String> {
        let row = self
            .schedule
            .row_for(&self.venue, self.segment, self.underlying.as_deref(), ts, false)
            .map_err(|e| e.to_string())?;
        self.note_used(row);
        Ok(row.rates)
    }

    fn note_used(&self, row: &crate::execution::costs::FeeRow) {
        if row.meta.status == crate::execution::costs::RowStatus::Unverified {
            self.unverified_set().insert(row.label());
        }
    }

    /// The start of the venue's trading date containing `ts`, in UTC ns:
    /// the instant whose row prices every fill of that date.
    fn day_start(&self, ts: i64) -> i64 {
        (ts + self.tz_offset_ns).div_euclid(DAY_NS) * DAY_NS - self.tz_offset_ns
    }

    /// Move to the trading date of `ts`. A date prices on one row, the one in
    /// force at its start, so the schedule is read when the date changes and
    /// not on every step.
    fn set_time(&mut self, ts: i64) {
        let day = (ts + self.tz_offset_ns).div_euclid(DAY_NS);
        if day == self.day {
            return;
        }
        if self.unresolved.is_none() {
            match self.rates_used_at(self.day_start(ts)) {
                Ok(r) => {
                    self.current = r;
                    self.gap = None;
                }
                Err(why) => self.gap = Some(why),
            }
        }
        self.day = day;
    }

    /// What earlier fills of the order in progress traded.
    fn order_before(&self) -> f64 {
        match self.order_key.load(Ordering::Relaxed) {
            0 => 0.0,
            key => self.turnover_map().get(&key).copied().unwrap_or(0.0),
        }
    }

    fn unverified_set(&self) -> std::sync::MutexGuard<'_, BTreeSet<String>> {
        self.unverified.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn slice_log(&self) -> std::sync::MutexGuard<'_, VecDeque<(u64, f64)>> {
        self.slices.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn turnover_map(&self) -> std::sync::MutexGuard<'_, HashMap<u64, f64>> {
        self.order_turnover.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn remember(&self, value: f64) {
        self.last_value.store(value.abs().to_bits(), Ordering::Relaxed);
    }

    fn day_fee_owed(&self) -> bool {
        self.paid_day.load(Ordering::Relaxed) != self.day
    }
}

/// Fee model for calculating transaction costs.
#[derive(Debug, Clone)]
pub enum FeeModel {
    /// No fees.
    None,
    /// Fixed percentage of trade value.
    Percentage(f64),
    /// Fixed fee per trade.
    Fixed(f64),
    /// Per-share/contract fee.
    PerShare(f64),
    /// Tiered fee structure based on trade value.
    Tiered(Vec<(f64, f64)>), // (threshold, rate)
    /// Custom fee function (stored as percentage for simplicity).
    Custom { base: f64, per_share: f64 },
    /// Itemized Indian regulatory costs for a market segment.
    ///
    /// Unlike the other variants this produces a per-component breakdown, so
    /// the equity curve and the reported costs are the same numbers.
    Indian { segment: Segment },
    /// Itemised costs from a row of a fee schedule: the rates a user loaded,
    /// or a built-in row resolved once for a venue and segment. Same
    /// computation as `Indian`, and the same breakdown.
    Rates(FeeRates),
    /// A fee schedule priced by the row in force on each fill's date, with
    /// settlement rows and the per-instrument-day fee. See [`DatedFees`].
    Dated(Box<DatedFees>),
}

/// A long entry buys and a long exit sells; a short is the reverse.
#[inline]
fn is_buy(direction: Direction, is_entry: bool) -> bool {
    matches!((direction, is_entry), (Direction::Long, true) | (Direction::Short, false))
}

impl Default for FeeModel {
    fn default() -> Self {
        FeeModel::Percentage(0.001) // 0.1% default
    }
}

impl FeeModel {
    /// Create a new percentage fee model.
    pub fn percentage(rate: f64) -> Self {
        FeeModel::Percentage(rate)
    }

    /// Create a new fixed fee model.
    pub fn fixed(amount: f64) -> Self {
        FeeModel::Fixed(amount)
    }

    /// Create a new per-share fee model.
    pub fn per_share(rate: f64) -> Self {
        FeeModel::PerShare(rate)
    }

    /// Calculate fee for a trade.
    ///
    /// # Arguments
    /// * `price` - Trade price
    /// * `size` - Position size (shares/contracts)
    /// * `direction` - Trade direction (for asymmetric fees if needed)
    ///
    /// # Returns
    /// Fee amount
    pub fn calculate(&self, price: Price, size: f64, _direction: Direction) -> f64 {
        let trade_value = price * size.abs();

        match self {
            FeeModel::None => 0.0,
            FeeModel::Percentage(rate) => trade_value * rate,
            FeeModel::Fixed(amount) => *amount,
            FeeModel::PerShare(rate) => size.abs() * rate,
            FeeModel::Tiered(tiers) => {
                // Find applicable tier
                let mut applicable_rate = 0.0;
                for (threshold, rate) in tiers {
                    if trade_value >= *threshold {
                        applicable_rate = *rate;
                    } else {
                        break;
                    }
                }
                trade_value * applicable_rate
            }
            FeeModel::Custom { base, per_share } => base + size.abs() * per_share,
            FeeModel::Indian { segment } => {
                segment.rates().charge(is_buy(_direction, true), trade_value).total()
            }
            FeeModel::Rates(rates) => rates.charge(is_buy(_direction, true), trade_value).total(),
            FeeModel::Dated(d) => {
                d.remember(trade_value);
                d.rates()
                    .charge_fill(
                        is_buy(_direction, true),
                        trade_value,
                        d.order_before(),
                        d.day_fee_owed(),
                    )
                    .total()
            }
        }
    }

    /// Itemized costs for one side of a trade.
    ///
    /// Returns `None` for the flat models, which have no component structure.
    /// Unlike [`FeeModel::calculate`], this distinguishes entry from exit,
    /// which matters because STT lands on the sell leg and stamp duty on the
    /// buy leg.
    pub fn breakdown(
        &self,
        price: Price,
        size: f64,
        direction: Direction,
        is_entry: bool,
    ) -> Option<FeeBreakdown> {
        match self {
            FeeModel::Indian { segment } => {
                Some(segment.rates().charge(is_buy(direction, is_entry), price * size.abs()))
            }
            FeeModel::Rates(rates) => {
                Some(rates.charge(is_buy(direction, is_entry), price * size.abs()))
            }
            FeeModel::Dated(d) => {
                d.remember(price * size);
                Some(d.rates().charge_fill(
                    is_buy(direction, is_entry),
                    (price * size).abs(),
                    d.order_before(),
                    d.day_fee_owed(),
                ))
            }
            _ => None,
        }
    }

    /// Move a dated model to the time of the step about to run, so its fills
    /// are priced on the row in force on their date. Other models ignore it.
    pub fn set_time(&mut self, ts: i64) {
        if let FeeModel::Dated(d) = self {
            d.set_time(ts);
        }
    }

    /// Attach the instrument a dated model prices: its underlying for rows
    /// that name one, and, when the run did not fix a row family, its venue
    /// and kind. `intraday` says positions are squared off the same day (cash
    /// equity is then priced as intraday). Other models ignore it.
    pub fn bind_instrument(&mut self, spec: &InstrumentSpec, intraday: bool) -> Result<(), String> {
        match self {
            FeeModel::Dated(d) => d.bind_instrument(spec, intraday),
            _ => Ok(()),
        }
    }

    /// A fill was booked with this breakdown: if it carried the per-day fee,
    /// the trading date has paid it.
    pub fn note_booked(&self, breakdown: Option<&FeeBreakdown>) {
        if let (FeeModel::Dated(d), Some(b)) = (self, breakdown) {
            if b.dp_charge > 0.0 {
                d.paid_day.store(d.day, Ordering::Relaxed);
            }
            let key = d.order_key.load(Ordering::Relaxed);
            if key != 0 {
                let value = f64::from_bits(d.last_value.load(Ordering::Relaxed));
                let mut map = d.turnover_map();
                let before = map.get(&key).copied().unwrap_or(0.0);
                map.insert(key, before + value);
                let mut log = d.slice_log();
                if log.len() >= RECENT_SLICES {
                    log.pop_front();
                }
                log.push_back((key - 1, before));
            }
        }
    }

    /// The fills about to be booked belong to order `id`: a brokerage the
    /// schedule charges per order is charged once over all of them. Call
    /// [`FeeModel::end_order`] when they are booked. Other models ignore it.
    pub fn begin_order(&self, id: u64) {
        if let FeeModel::Dated(d) = self {
            d.order_key.store(id + 1, Ordering::Relaxed);
        }
    }

    /// The fills of the order are booked. `live` says which orders can still
    /// fill; the turnover kept for any other order is dropped, so the memory
    /// held is bounded by the orders working with a partial fill.
    pub fn end_order(&self, live: impl Fn(u64) -> bool) {
        if let FeeModel::Dated(d) = self {
            d.order_key.store(0, Ordering::Relaxed);
            d.turnover_map().retain(|key, _| live(key - 1));
        }
    }

    /// What the earlier fills of `id` had traded when its oldest unpriced
    /// booked fill was booked (0 for none), consumed. A venue simulation that
    /// prices a fill after the kernel booked it uses this so its fee is the
    /// booked one, brokerage per order and all.
    pub fn take_slice_before(&self, id: u64) -> f64 {
        let FeeModel::Dated(d) = self else { return 0.0 };
        let mut log = d.slice_log();
        match log.iter().position(|(order, _)| *order == id) {
            Some(at) => log.remove(at).map_or(0.0, |(_, before)| before),
            None => 0.0,
        }
    }

    /// Itemised costs as [`FeeModel::breakdown`], for a fill whose order had
    /// already traded `order_before` (see [`FeeModel::take_slice_before`]).
    pub fn breakdown_after(
        &self,
        order_before: f64,
        price: Price,
        size: f64,
        direction: Direction,
        is_entry: bool,
    ) -> Option<FeeBreakdown> {
        match self {
            FeeModel::Dated(d) => Some(d.rates().charge_fill(
                is_buy(direction, is_entry),
                (price * size).abs(),
                order_before,
                d.day_fee_owed(),
            )),
            _ => self.breakdown(price, size, direction, is_entry),
        }
    }

    /// The labels (`VENUE:SEGMENT[:KEY]`) of the unverified rows fills were
    /// priced on, sorted.
    pub fn unverified_rows(&self) -> Vec<String> {
        match self {
            FeeModel::Dated(d) => d.unverified_set().iter().cloned().collect(),
            _ => Vec::new(),
        }
    }

    /// Orders whose turnover is being kept (for tests of the bound).
    #[cfg(test)]
    pub(crate) fn tracked_orders(&self) -> (usize, usize) {
        match self {
            FeeModel::Dated(d) => (d.turnover_map().len(), d.slice_log().len()),
            _ => (0, 0),
        }
    }

    /// The charges of a round trip of cash equity opened and closed on one trading date (the
    /// venue's own, by its session time zone), repriced on the intraday row of that date:
    /// `(entry, exit)` for `entry_value` and `exit_value` traded by a `direction` line. `None`
    /// when the model does not classify by day (not a dated schedule, a row family the run
    /// fixed, an instrument that is not cash equity, a schedule without an intraday row) or
    /// the legs fall on different dates, in which case the delivery charges stand.
    ///
    /// The kernel prices an entry as delivery because it cannot know the exit; this is the
    /// correction when the exit lands the same day. Only what differs between the two rows is
    /// restated: the brokerage stays as it was booked (`entry_brokerage`, `exit_brokerage`), so an
    /// order a card charges per order is not charged again on each slice of a partly closed
    /// position.
    pub fn same_day_charges(
        &self,
        entry_value: f64,
        exit_value: f64,
        direction: Direction,
        entry_ts: i64,
        exit_ts: i64,
        brokerage: SameDayBrokerage,
    ) -> Option<(FeeBreakdown, FeeBreakdown)> {
        let FeeModel::Dated(d) = self else { return None };
        if !d.classify_by_day || d.unresolved.is_some() {
            return None;
        }
        let day = d.day_start(entry_ts);
        if day != d.day_start(exit_ts) {
            return None;
        }
        let row = match d.schedule.row_for(
            &d.venue,
            FeeSegment::Intraday,
            d.underlying.as_deref(),
            day,
            false,
        ) {
            Ok(row) => row,
            Err(why) => fee_gap_fault(why),
        };
        d.note_used(row);
        let entry_buy = is_buy(direction, true);
        let entry_brokerage = match brokerage.entry {
            Some(b) => b,
            None => {
                d.booked_entry_brokerage(day, entry_buy, entry_value.abs(), brokerage.entry_total)
            }
        };
        Some((
            row.rates.charge_keeping_brokerage(entry_buy, entry_value, entry_brokerage),
            row.rates.charge_keeping_brokerage(
                is_buy(direction, false),
                exit_value,
                brokerage.exit,
            ),
        ))
    }

    /// The charge on settling `size` contracts at `price` per contract
    /// (already times the multiplier) at `ts`, on the schedule's settlement
    /// row in force then, for the side closing a `direction` line. `None`
    /// when the model has no settlement row to price it (the caller falls
    /// back to the instrument's `settlement_fee`). A settlement row that
    /// exists but not for that date stops with [`fee_gap_fault`].
    pub fn settlement_breakdown(
        &self,
        price: Price,
        size: f64,
        direction: Direction,
        ts: i64,
    ) -> Option<FeeBreakdown> {
        let FeeModel::Dated(d) = self else { return None };
        if d.unresolved.is_some() {
            return None;
        }
        let row =
            match d.schedule.row_for_opt(&d.venue, d.segment, d.underlying.as_deref(), ts, true) {
                Ok(Some(row)) => row,
                Ok(None) => return None,
                Err(why) => fee_gap_fault(why),
            };
        d.note_used(row);
        Some(row.rates.charge_fill(is_buy(direction, false), (price * size).abs(), 0.0, false))
    }

    /// The charge on taking or giving physical delivery of the underlying
    /// stock of a stock future or an exercised stock option, as the delivery
    /// leg of its own: the cash market's delivery rates (STT on both sides,
    /// stamp duty on the buyer), whatever segment the derivative is priced in,
    /// so the derivative's trade STT is not stacked on it. `price` is the
    /// delivery value per unit (times the multiplier); `receives` is true when
    /// the stock is taken in (a buy). `None` when the model is not an exchange
    /// schedule, in which case the caller prices the leg as any fill. A dated
    /// schedule with no delivery row for the instrument stops with the reason. The
    /// depository charge, owed when delivered stock is later sold, is not part
    /// of the delivery.
    pub fn delivery_breakdown(
        &self,
        price: Price,
        size: f64,
        receives: bool,
        ts: i64,
    ) -> Option<FeeBreakdown> {
        let value = (price * size).abs();
        match self {
            FeeModel::Dated(d) => {
                let cash = match d.venue.as_str() {
                    "NFO" => "NSE",
                    "BFO" => "BSE",
                    other => other,
                };
                // BSE cash charges depend on the scrip group, which the derivative's spec carries.
                let key = if cash == "BSE" { d.scrip_group.as_deref() } else { None };
                // A delivery that cannot be priced is never priced as something else: the
                // bindings refuse such an instrument when it is registered (a physical
                // derivative whose delivery row is missing or needs a scrip group), so a
                // failure here is a broken invariant and stops with its reason.
                let row = d
                    .schedule
                    .row_for(cash, FeeSegment::Delivery, key, ts, false)
                    .unwrap_or_else(|e| {
                        panic!(
                            "fee schedule {:?} cannot price a delivery on {cash}: {e}",
                            d.schedule.name
                        )
                    });
                d.note_used(row);
                Some(row.rates.charge_fill(receives, value, 0.0, false))
            }
            FeeModel::Indian { .. } => {
                let rates = Segment::EquityDelivery.rates();
                Some(rates.charge_fill(receives, value, 0.0, false))
            }
            _ => None,
        }
    }

    /// Fee for one side, honoring entry/exit asymmetry where the model has any.
    pub fn calculate_side(
        &self,
        price: Price,
        size: f64,
        direction: Direction,
        is_entry: bool,
    ) -> f64 {
        match self.breakdown(price, size, direction, is_entry) {
            Some(b) => b.total(),
            None => self.calculate(price, size, direction),
        }
    }

    /// Create an itemized Indian cost model for a segment.
    pub fn indian(segment: Segment) -> Self {
        FeeModel::Indian { segment }
    }

    /// Calculate round-trip fees (entry + exit).
    pub fn round_trip(
        &self,
        entry_price: Price,
        exit_price: Price,
        size: f64,
        direction: Direction,
    ) -> f64 {
        self.calculate(entry_price, size, direction) + self.calculate(exit_price, size, direction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentage_fee() {
        let fee = FeeModel::percentage(0.001);
        let result = fee.calculate(100.0, 100.0, Direction::Long);
        assert!((result - 10.0).abs() < 1e-10); // 100 * 100 * 0.001 = 10
    }

    #[test]
    fn test_fixed_fee() {
        let fee = FeeModel::fixed(5.0);
        let result = fee.calculate(100.0, 100.0, Direction::Long);
        assert!((result - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_per_share_fee() {
        let fee = FeeModel::per_share(0.01);
        let result = fee.calculate(100.0, 100.0, Direction::Long);
        assert!((result - 1.0).abs() < 1e-10); // 100 * 0.01 = 1
    }

    #[test]
    fn test_round_trip() {
        let fee = FeeModel::percentage(0.001);
        let result = fee.round_trip(100.0, 110.0, 100.0, Direction::Long);
        // Entry: 100 * 100 * 0.001 = 10
        // Exit: 110 * 100 * 0.001 = 11
        // Total: 21
        assert!((result - 21.0).abs() < 1e-10);
    }

    #[test]
    fn test_no_fee() {
        let fee = FeeModel::None;
        let result = fee.calculate(100.0, 100.0, Direction::Long);
        assert!((result - 0.0).abs() < 1e-10);
    }

    /// The itemized model splits a round trip across its two sides.
    ///
    /// `calculate` alone always prices the entry side, so an exit priced
    /// through it would carry the wrong side-specific charges. These cover
    /// `breakdown`/`calculate_side`, which the flat-rate tests above cannot
    /// reach.
    #[test]
    fn indian_model_charges_each_side_separately() {
        use crate::execution::indian_costs::Segment;
        let model = FeeModel::indian(Segment::OptionsNfo);

        let entry = model.breakdown(100.0, 75.0, Direction::Long, true).unwrap();
        let exit = model.breakdown(100.0, 75.0, Direction::Long, false).unwrap();

        // A long buys to open: stamp duty on entry, transaction tax on exit.
        assert!(entry.stamp_duty > 0.0);
        assert_eq!(entry.stt, 0.0);
        assert_eq!(exit.stamp_duty, 0.0);
        assert!(exit.stt > 0.0);

        // The exchange charge lands on both sides; a built-in row has no brokerage.
        assert_eq!(entry.exchange_txn, exit.exchange_txn);
        assert!(entry.exchange_txn > 0.0);
        assert_eq!((entry.brokerage, exit.brokerage), (0.0, 0.0));
    }

    /// `calculate_side` returns the itemized total, or the flat fee when the
    /// model has no component structure.
    #[test]
    fn calculate_side_matches_the_breakdown_it_reports() {
        use crate::execution::indian_costs::Segment;
        let indian = FeeModel::indian(Segment::OptionsNfo);
        let side = indian.calculate_side(100.0, 75.0, Direction::Short, true);
        let breakdown = indian.breakdown(100.0, 75.0, Direction::Short, true).unwrap();
        assert!((side - breakdown.total()).abs() < 1e-9);

        // A flat model has no breakdown to report, and falls back cleanly.
        let flat = FeeModel::percentage(0.001);
        assert!(flat.breakdown(100.0, 75.0, Direction::Long, true).is_none());
        assert_eq!(flat.calculate_side(100.0, 75.0, Direction::Long, true), 7.5);
    }

    /// The order bookkeeping is bounded: a long run of orders leaves only the
    /// orders still working with a partial fill, and a fixed log of recent fills.
    #[test]
    fn the_order_bookkeeping_stays_bounded_over_a_long_run() {
        use crate::execution::costs::{BrokeragePer, FeeRow};
        let rates =
            FeeRates { brokerage_flat: 20.0, brokerage_per: BrokeragePer::Order, ..FeeRates::ZERO };
        let row = FeeRow {
            venue: "NFO".into(),
            segment: FeeSegment::Options,
            underlying: None,
            settlement: false,
            effective_from: i64::MIN,
            rates,
            meta: Default::default(),
        };
        let schedule = Arc::new(FeeSchedule::new("t", "t", vec![row]));
        let dated = DatedFees::new(schedule, "NFO".into(), FeeSegment::Options, 0).unwrap();
        let model = FeeModel::Dated(Box::new(dated));
        for id in 1..=50_000u64 {
            model.begin_order(id);
            let b = model.breakdown(100.0, 1.0, Direction::Long, true).unwrap();
            model.note_booked(Some(&b));
            // order 1 stays working with a partial fill; every other order completes
            model.end_order(|live| live == 1);
        }
        let (orders, log) = model.tracked_orders();
        assert_eq!(orders, 1, "only the order still working is tracked");
        assert!(log <= RECENT_SLICES, "the fill log is capped, got {log}");
    }

    /// A brokerage a card charges per order is charged once over the order's
    /// fills, however many slices, and again for the next order.
    #[test]
    fn per_order_brokerage_is_charged_once_over_an_orders_slices() {
        use crate::execution::costs::{BrokeragePer, FeeRow};
        let rates =
            FeeRates { brokerage_flat: 20.0, brokerage_per: BrokeragePer::Order, ..FeeRates::ZERO };
        let row = FeeRow {
            venue: "NFO".into(),
            segment: FeeSegment::Options,
            underlying: None,
            settlement: false,
            effective_from: i64::MIN,
            rates,
            meta: Default::default(),
        };
        let schedule = Arc::new(FeeSchedule::new("t", "t", vec![row]));
        let dated = DatedFees::new(schedule, "NFO".into(), FeeSegment::Options, 0).unwrap();
        let model = FeeModel::Dated(Box::new(dated));
        let slice = |m: &FeeModel| {
            let b = m.breakdown(100.0, 10.0, Direction::Long, true).unwrap();
            m.note_booked(Some(&b));
            b.brokerage
        };
        let working = |id: u64| id == 7;
        model.begin_order(7);
        assert_eq!(slice(&model), 20.0);
        model.end_order(working);
        model.begin_order(7);
        assert_eq!(slice(&model), 0.0, "the second slice of the same order owes none");
        model.end_order(|_| false);
        model.begin_order(8);
        assert_eq!(slice(&model), 20.0, "the next order owes its own");
        model.end_order(|_| false);
        // a venue simulation pricing the booked fills afterwards gets the same fees
        let priced = |before: f64| {
            model.breakdown_after(before, 100.0, 10.0, Direction::Long, true).unwrap().brokerage
        };
        assert_eq!(priced(model.take_slice_before(7)), 20.0, "the first slice of order 7");
        assert_eq!(priced(model.take_slice_before(7)), 0.0, "the second slice of order 7");
        assert_eq!(priced(model.take_slice_before(8)), 20.0);
        assert_eq!(model.breakdown(100.0, 10.0, Direction::Long, true).unwrap().brokerage, 20.0);
    }

    fn dated_row(settlement: bool, from: i64) -> crate::execution::costs::FeeRow {
        crate::execution::costs::FeeRow {
            venue: "NFO".into(),
            segment: FeeSegment::Options,
            underlying: None,
            settlement,
            effective_from: from,
            rates: FeeRates { tax_sell_rate: 0.001, ..FeeRates::ZERO },
            meta: Default::default(),
        }
    }

    fn dated_model(rows: Vec<crate::execution::costs::FeeRow>) -> FeeModel {
        let schedule = Arc::new(FeeSchedule::new("t", "t", rows));
        FeeModel::Dated(Box::new(
            DatedFees::new(schedule, "NFO".into(), FeeSegment::Options, 0).unwrap(),
        ))
    }

    /// A schedule with no settlement row has nothing to charge at expiry; that is not a lookup
    /// failure, and the caller falls back to the instrument's own fee.
    #[test]
    fn a_schedule_without_a_settlement_row_has_nothing_to_charge() {
        let model = dated_model(vec![dated_row(false, 1_000)]);
        assert!(model.settlement_breakdown(100.0, 1.0, Direction::Long, 5_000).is_none());
    }

    /// A settlement row that exists but not yet for the date is a lookup error, never `None`.
    #[test]
    #[should_panic(expected = "fee_schedule_gap:")]
    fn a_settlement_before_the_first_settlement_row_stops_with_the_gap() {
        let model = dated_model(vec![dated_row(false, 1_000), dated_row(true, 2_000)]);
        let _ = model.settlement_breakdown(100.0, 1.0, Direction::Long, 1_500);
    }

    /// A fill on a date before the first row stops with the gap; a date with no fill does not.
    #[test]
    #[should_panic(expected = "earliest row takes effect on")]
    fn a_fill_before_the_first_row_stops_with_the_gap() {
        let mut model = dated_model(vec![dated_row(false, 10 * DAY_NS)]);
        model.set_time(DAY_NS);
        let _ = model.breakdown(100.0, 1.0, Direction::Long, true);
    }

    #[test]
    fn a_date_before_the_first_row_without_a_fill_charges_nothing_and_stops_nothing() {
        let mut model = dated_model(vec![dated_row(false, 10 * DAY_NS)]);
        model.set_time(DAY_NS);
        model.set_time(11 * DAY_NS);
        assert!(model.breakdown(100.0, 1.0, Direction::Short, false).is_some());
    }
}
