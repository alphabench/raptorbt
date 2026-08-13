//! Itemized Indian transaction costs.
//!
//! The engine charges fees per side (once on entry, once on exit), so every
//! rate here is applied to one leg. Regulatory schedules are usually quoted
//! per round trip -- STT "on sell side", stamp duty "on buy side" -- which is
//! why each component below states which side it lands on.
//!
//! Rates follow the published Zerodha/NSE schedules, and the engine charges
//! them directly so its equity curve and the reported cost breakdown describe
//! the same money. Recomputing an itemized breakdown separately for display
//! leaves the two free to disagree.

use crate::core::types::Direction;
use serde::{Deserialize, Serialize};

/// Market segment and instrument type, selecting a cost schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Segment {
    /// NSE/BSE equity, squared off same day.
    EquityIntraday,
    /// NSE/BSE equity, held overnight.
    EquityDelivery,
    /// NFO/BFO index and stock futures.
    FuturesNfo,
    /// NFO/BFO options. STT and exchange charges apply to premium.
    OptionsNfo,
    /// MCX commodity futures. No STT.
    FuturesMcx,
    /// MCX commodity options.
    OptionsMcx,
    /// CDS currency futures. No STT.
    FuturesCds,
    /// CDS currency options. No STT.
    OptionsCds,
}

/// Which side of a round trip a charge applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChargedOn {
    /// Sell leg only.
    Sell,
    /// Both legs.
    Both,
    /// Not charged.
    Never,
}

/// Regulatory rates for one segment. All rates are fractions.
#[derive(Debug, Clone, Copy)]
pub struct CostSchedule {
    /// Flat brokerage per executed order, in rupees.
    pub brokerage_per_order: f64,
    /// Securities Transaction Tax rate.
    pub stt_rate: f64,
    /// Which leg STT applies to.
    stt_on: ChargedOn,
    /// Exchange transaction charge rate.
    pub exchange_txn_rate: f64,
    /// SEBI turnover fee (about Rs 10 per crore).
    pub sebi_turnover_rate: f64,
    /// Stamp duty rate, charged on the buy leg only.
    pub stamp_duty_rate: f64,
    /// GST on brokerage + exchange + SEBI charges.
    pub gst_rate: f64,
}

/// Itemized costs for one side of a trade.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct FeeBreakdown {
    pub brokerage: f64,
    pub stt: f64,
    pub exchange_txn: f64,
    pub sebi_fee: f64,
    pub stamp_duty: f64,
    pub gst: f64,
}

impl FeeBreakdown {
    /// Sum of all components.
    #[inline]
    pub fn total(&self) -> f64 {
        self.brokerage + self.stt + self.exchange_txn + self.sebi_fee + self.stamp_duty + self.gst
    }

    /// Accumulate another breakdown, for aggregating across trades.
    pub fn add(&mut self, other: &FeeBreakdown) {
        self.brokerage += other.brokerage;
        self.stt += other.stt;
        self.exchange_txn += other.exchange_txn;
        self.sebi_fee += other.sebi_fee;
        self.stamp_duty += other.stamp_duty;
        self.gst += other.gst;
    }
}

const GST: f64 = 0.18;
const SEBI: f64 = 0.000001;
const BROKERAGE: f64 = 20.0;

/// Depository participant charge on equity delivery sells, in rupees:
/// Rs 13.00 (Zerodha DP fee, CDSL) + 18% GST = Rs 15.34.
///
/// This is **per ISIN per day with any sell**, not per order -- selling one
/// scrip across five orders in a day incurs it once, selling five scrips
/// incurs it five times. That unit is why it is NOT a `CostSchedule` field:
/// every rate there applies per side of one trade, and folding a per-ISIN-day
/// flat fee into them would either overcount multi-order days or undercount
/// multi-scrip days. Consumers (the rebalance simulator, and any small-book
/// refusal arithmetic built on it) count distinct ISINs with net sells per day
/// and multiply.
///
/// This flat charge, not the percentage rates, is what dominates rebalancing
/// costs on small delivery books -- at a Rs 3,000 sell it is ~0.5% before STT.
pub const DP_SELL_CHARGE_PER_ISIN_PER_DAY: f64 = 15.34;

impl Segment {
    /// Regulatory schedule for this segment.
    pub fn schedule(&self) -> CostSchedule {
        match self {
            Segment::EquityIntraday => CostSchedule {
                brokerage_per_order: BROKERAGE,
                stt_rate: 0.00025,
                stt_on: ChargedOn::Sell,
                exchange_txn_rate: 0.0000345,
                sebi_turnover_rate: SEBI,
                stamp_duty_rate: 0.00003,
                gst_rate: GST,
            },
            Segment::EquityDelivery => CostSchedule {
                brokerage_per_order: BROKERAGE,
                stt_rate: 0.001,
                stt_on: ChargedOn::Both,
                exchange_txn_rate: 0.0000345,
                sebi_turnover_rate: SEBI,
                stamp_duty_rate: 0.00015,
                gst_rate: GST,
            },
            Segment::FuturesNfo => CostSchedule {
                brokerage_per_order: BROKERAGE,
                stt_rate: 0.0001,
                stt_on: ChargedOn::Sell,
                exchange_txn_rate: 0.00002,
                sebi_turnover_rate: SEBI,
                stamp_duty_rate: 0.00002,
                gst_rate: GST,
            },
            Segment::OptionsNfo => CostSchedule {
                brokerage_per_order: BROKERAGE,
                stt_rate: 0.000625,
                stt_on: ChargedOn::Sell,
                exchange_txn_rate: 0.00035,
                sebi_turnover_rate: SEBI,
                stamp_duty_rate: 0.00003,
                gst_rate: GST,
            },
            Segment::FuturesMcx => CostSchedule {
                brokerage_per_order: BROKERAGE,
                stt_rate: 0.0,
                stt_on: ChargedOn::Never,
                exchange_txn_rate: 0.00002,
                sebi_turnover_rate: SEBI,
                stamp_duty_rate: 0.00002,
                gst_rate: GST,
            },
            Segment::OptionsMcx => CostSchedule {
                brokerage_per_order: BROKERAGE,
                stt_rate: 0.0005,
                stt_on: ChargedOn::Sell,
                exchange_txn_rate: 0.00035,
                sebi_turnover_rate: SEBI,
                stamp_duty_rate: 0.00003,
                gst_rate: GST,
            },
            Segment::FuturesCds => CostSchedule {
                brokerage_per_order: BROKERAGE,
                stt_rate: 0.0,
                stt_on: ChargedOn::Never,
                exchange_txn_rate: 0.0000035,
                sebi_turnover_rate: SEBI,
                stamp_duty_rate: 0.00001,
                gst_rate: GST,
            },
            Segment::OptionsCds => CostSchedule {
                brokerage_per_order: BROKERAGE,
                stt_rate: 0.0,
                stt_on: ChargedOn::Never,
                exchange_txn_rate: 0.00031,
                sebi_turnover_rate: SEBI,
                stamp_duty_rate: 0.00003,
                gst_rate: GST,
            },
        }
    }

    /// Whether STT and exchange charges apply to option premium rather than
    /// contract notional.
    #[inline]
    pub fn charges_on_premium(&self) -> bool {
        matches!(self, Segment::OptionsNfo | Segment::OptionsMcx | Segment::OptionsCds)
    }

    /// Parse a segment name, as it appears on a broker's instrument records.
    ///
    /// `intraday` distinguishes equity intraday from delivery; it is ignored
    /// for derivative segments, which are always intraday-rated.
    pub fn parse(segment: &str, instrument_type: Option<&str>, intraday: bool) -> Option<Self> {
        let seg = segment.to_ascii_uppercase();
        let ty = instrument_type.map(|t| t.to_ascii_uppercase());
        let is_option = matches!(ty.as_deref(), Some("OPT") | Some("CE") | Some("PE"));

        match seg.as_str() {
            "NSE" | "BSE" => {
                Some(if intraday { Segment::EquityIntraday } else { Segment::EquityDelivery })
            }
            "NFO" | "BFO" => {
                Some(if is_option { Segment::OptionsNfo } else { Segment::FuturesNfo })
            }
            "MCX" => Some(if is_option { Segment::OptionsMcx } else { Segment::FuturesMcx }),
            "CDS" | "BCD" => {
                Some(if is_option { Segment::OptionsCds } else { Segment::FuturesCds })
            }
            _ => None,
        }
    }
}

/// Itemized cost for **one side** of a trade.
///
/// `value` is the notional for that leg; for options it is the premium, since
/// STT and exchange charges are levied on premium there.
///
/// The engine charges entry and exit separately, so each side-specific charge
/// (STT on sell, stamp duty on buy) lands only on the leg that owes it -- not
/// halved across both, which is how a round-trip formula expresses the same
/// thing.
pub fn calculate_side(
    segment: Segment,
    value: f64,
    direction: Direction,
    is_entry: bool,
) -> FeeBreakdown {
    let schedule = segment.schedule();
    let value = value.abs();

    // A long entry buys and a long exit sells; a short is the reverse.
    let is_buy =
        matches!((direction, is_entry), (Direction::Long, true) | (Direction::Short, false));

    let mut fees = FeeBreakdown { brokerage: schedule.brokerage_per_order, ..Default::default() };

    fees.stt = match schedule.stt_on {
        ChargedOn::Never => 0.0,
        ChargedOn::Both => schedule.stt_rate * value,
        ChargedOn::Sell => {
            if is_buy {
                0.0
            } else {
                schedule.stt_rate * value
            }
        }
    };

    fees.exchange_txn = schedule.exchange_txn_rate * value;
    fees.sebi_fee = schedule.sebi_turnover_rate * value;
    fees.stamp_duty = if is_buy { schedule.stamp_duty_rate * value } else { 0.0 };

    // GST applies to brokerage and exchange-side charges, never to STT or
    // stamp duty (themselves taxes).
    fees.gst = schedule.gst_rate * (fees.brokerage + fees.exchange_txn + fees.sebi_fee);

    fees
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A round trip should charge STT once (sell leg) and stamp duty once (buy).
    #[test]
    fn round_trip_charges_each_side_specific_component_once() {
        let value = 1_000_000.0;
        let entry = calculate_side(Segment::EquityIntraday, value, Direction::Long, true);
        let exit = calculate_side(Segment::EquityIntraday, value, Direction::Long, false);

        assert_eq!(entry.stt, 0.0, "no STT on a long entry (buy)");
        assert!(exit.stt > 0.0, "STT lands on the long exit (sell)");
        assert!(entry.stamp_duty > 0.0, "stamp duty lands on the buy");
        assert_eq!(exit.stamp_duty, 0.0, "no stamp duty on the sell");

        // Matches the round-trip figure: 0.025% of one leg.
        assert!((exit.stt - 0.00025 * value).abs() < 1e-9);
    }

    /// Shorts reverse which leg is the buy.
    #[test]
    fn short_direction_reverses_buy_and_sell_legs() {
        let value = 1_000_000.0;
        let entry = calculate_side(Segment::EquityIntraday, value, Direction::Short, true);
        let exit = calculate_side(Segment::EquityIntraday, value, Direction::Short, false);

        assert!(entry.stt > 0.0, "a short entry is a sell, so STT applies");
        assert_eq!(exit.stt, 0.0, "the covering buy owes no STT");
        assert_eq!(entry.stamp_duty, 0.0);
        assert!(exit.stamp_duty > 0.0, "stamp duty lands on the covering buy");
    }

    #[test]
    fn delivery_charges_stt_on_both_legs() {
        let value = 500_000.0;
        let entry = calculate_side(Segment::EquityDelivery, value, Direction::Long, true);
        let exit = calculate_side(Segment::EquityDelivery, value, Direction::Long, false);

        assert!((entry.stt - 0.001 * value).abs() < 1e-9);
        assert!((exit.stt - 0.001 * value).abs() < 1e-9);
    }

    #[test]
    fn commodity_and_currency_futures_have_no_stt() {
        for segment in [Segment::FuturesMcx, Segment::FuturesCds, Segment::OptionsCds] {
            let fees = calculate_side(segment, 1_000_000.0, Direction::Long, false);
            assert_eq!(fees.stt, 0.0, "{segment:?} should not levy STT");
        }
    }

    #[test]
    fn gst_excludes_stt_and_stamp_duty() {
        let fees = calculate_side(Segment::EquityDelivery, 1_000_000.0, Direction::Long, true);
        let expected = 0.18 * (fees.brokerage + fees.exchange_txn + fees.sebi_fee);
        assert!((fees.gst - expected).abs() < 1e-9, "GST must not be levied on taxes");
    }

    #[test]
    fn total_is_the_sum_of_components() {
        let fees = calculate_side(Segment::OptionsNfo, 250_000.0, Direction::Long, false);
        let manual = fees.brokerage
            + fees.stt
            + fees.exchange_txn
            + fees.sebi_fee
            + fees.stamp_duty
            + fees.gst;
        assert!((fees.total() - manual).abs() < 1e-12);
    }

    #[test]
    fn breakdown_accumulates() {
        let a = calculate_side(Segment::EquityIntraday, 100_000.0, Direction::Long, true);
        let b = calculate_side(Segment::EquityIntraday, 100_000.0, Direction::Long, false);
        let mut sum = a;
        sum.add(&b);
        assert!((sum.total() - (a.total() + b.total())).abs() < 1e-12);
    }

    #[test]
    fn segment_parsing_covers_broker_names() {
        assert_eq!(Segment::parse("NSE", None, true), Some(Segment::EquityIntraday));
        assert_eq!(Segment::parse("NSE", None, false), Some(Segment::EquityDelivery));
        assert_eq!(Segment::parse("nfo", Some("OPT"), true), Some(Segment::OptionsNfo));
        assert_eq!(Segment::parse("NFO", Some("FUT"), true), Some(Segment::FuturesNfo));
        assert_eq!(Segment::parse("MCX", Some("CE"), true), Some(Segment::OptionsMcx));
        assert_eq!(Segment::parse("CDS", Some("FUT"), true), Some(Segment::FuturesCds));
        assert_eq!(Segment::parse("UNKNOWN", None, true), None);
    }

    /// Options levy STT and exchange charges on premium, which is far smaller
    /// than contract notional -- the caller must pass premium, and this test
    /// documents the magnitude of getting it wrong.
    #[test]
    fn options_charges_scale_with_the_value_passed() {
        let premium = 50_000.0;
        let notional = 5_000_000.0;
        let on_premium = calculate_side(Segment::OptionsNfo, premium, Direction::Long, false);
        let on_notional = calculate_side(Segment::OptionsNfo, notional, Direction::Long, false);
        assert!(on_notional.total() > on_premium.total() * 50.0);
        assert!(Segment::OptionsNfo.charges_on_premium());
    }
}
