//! Fee calculation models.

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

use crate::core::types::{Direction, Price};
use crate::execution::costs::{FeeBreakdown, FeeRates, FeeSchedule, FeeSegment};
use crate::execution::indian_costs::Segment;

/// No trading date has paid the per-day fee yet.
const NO_DAY: i64 = i64::MIN;

const DAY_NS: i64 = 86_400_000_000_000;

/// The schedule, venue and segment a run is priced on; the kernels each build
/// their [`DatedFees`] from it.
#[derive(Debug, Clone)]
pub struct FeeBinding {
    pub schedule: Arc<FeeSchedule>,
    pub venue: String,
    pub segment: FeeSegment,
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
    current: FeeRates,
    /// Trading date of the step in progress.
    day: i64,
    /// Trading date that last paid the per-day fee.
    paid_day: AtomicI64,
}

impl Clone for DatedFees {
    fn clone(&self) -> Self {
        Self {
            schedule: Arc::clone(&self.schedule),
            venue: self.venue.clone(),
            segment: self.segment,
            underlying: self.underlying.clone(),
            tz_offset_ns: self.tz_offset_ns,
            current: self.current,
            day: self.day,
            paid_day: AtomicI64::new(self.paid_day.load(Ordering::Relaxed)),
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
            current,
            day: NO_DAY,
            paid_day: AtomicI64::new(NO_DAY),
        })
    }

    /// The row in force at `ts` (UTC ns), for the bound underlying.
    fn rates_at(&self, ts: i64) -> Option<FeeRates> {
        self.schedule
            .row_for(&self.venue, self.segment, self.underlying.as_deref(), ts, false)
            .ok()
            .map(|r| r.rates)
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
        if let Some(r) = self.rates_at(self.day_start(ts)) {
            self.current = r;
        }
        self.day = day;
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
            FeeModel::Dated(d) => d
                .current
                .charge_fill(is_buy(_direction, true), trade_value, 0.0, d.day_fee_owed())
                .total(),
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
            FeeModel::Dated(d) => Some(d.current.charge_fill(
                is_buy(direction, is_entry),
                (price * size).abs(),
                0.0,
                d.day_fee_owed(),
            )),
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

    /// Name the underlying a dated model's rows are looked up for.
    pub fn bind_underlying(&mut self, underlying: Option<&str>) {
        if let FeeModel::Dated(d) = self {
            d.underlying = underlying.map(str::to_owned);
            if let Some(r) = d.rates_at(i64::MAX) {
                d.current = r;
            }
            // the next step reads the row of its date for this underlying
            d.day = NO_DAY;
        }
    }

    /// A fill was booked with this breakdown: if it carried the per-day fee,
    /// the trading date has paid it.
    pub fn note_booked(&self, breakdown: Option<&FeeBreakdown>) {
        if let (FeeModel::Dated(d), Some(b)) = (self, breakdown) {
            if b.dp_charge > 0.0 {
                d.paid_day.store(d.day, Ordering::Relaxed);
            }
        }
    }

    /// The charge on settling `size` contracts at `price` per contract
    /// (already times the multiplier) at `ts`, on the schedule's settlement
    /// row in force then, for the side closing a `direction` line. `None`
    /// when the model has no settlement row to price it (the caller falls
    /// back to the instrument's `settlement_fee`).
    pub fn settlement_breakdown(
        &self,
        price: Price,
        size: f64,
        direction: Direction,
        ts: i64,
    ) -> Option<FeeBreakdown> {
        let FeeModel::Dated(d) = self else { return None };
        let row =
            d.schedule.row_for(&d.venue, d.segment, d.underlying.as_deref(), ts, true).ok()?;
        Some(row.rates.charge_fill(is_buy(direction, false), (price * size).abs(), 0.0, false))
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

        // Per-order brokerage lands on both sides.
        assert_eq!(entry.brokerage, exit.brokerage);
        assert!(entry.brokerage > 0.0);
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
}
