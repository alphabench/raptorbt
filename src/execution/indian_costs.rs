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
//!
//! Every rate below was verified against <https://zerodha.com/charges/> on
//! 2026-08-20 and is pinned field-by-field in this module's tests. Two
//! deliberate approximations, both stated here rather than hidden:
//!
//! * BSE equity and BFO derivatives share the NSE-family schedules. Their
//!   published exchange-transaction rates differ by under 0.2 bps, and
//!   execution truth for this platform is Zerodha on NSE/NFO.
//! * The STT levied on options that are *exercised* (0.15% of intrinsic) is
//!   not modelled -- the engine closes positions by trading out, and the
//!   settlement path freezes legs at their settlement value without
//!   simulating exercise.

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
    /// MCX commodity futures. CTT (0.01% non-agri) on the sell side.
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
    /// Flat brokerage per executed order, in rupees. When `brokerage_rate` is
    /// non-zero this is the cap ("0.03% or Rs 20, whichever is lower");
    /// zero means the broker charges nothing on this segment (equity
    /// delivery).
    pub brokerage_flat: f64,
    /// Percentage brokerage per executed order, applied to the order's value.
    /// Zero means the flat amount alone applies (options; delivery).
    pub brokerage_rate: f64,
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

impl CostSchedule {
    /// Brokerage owed on one executed order of `value` rupees.
    ///
    /// Zerodha's schedule is "0.03% or Rs 20 per executed order, whichever is
    /// lower" on intraday equity and all futures, flat Rs 20 on options, and
    /// zero on equity delivery. A single flat figure cannot express that: the
    /// old unconditional Rs 20 overcharged every delivery trade (Zerodha
    /// charges nothing there) and every small intraday/futures order (below
    /// about Rs 66,667 the percentage is cheaper).
    #[inline]
    pub fn brokerage_for_order(&self, value: f64) -> f64 {
        if self.brokerage_rate > 0.0 {
            (self.brokerage_rate * value).min(self.brokerage_flat)
        } else {
            self.brokerage_flat
        }
    }
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
/// Flat brokerage per executed order (also the cap where a rate applies).
const BROKERAGE: f64 = 20.0;
/// "0.03% or Rs 20, whichever is lower" -- intraday equity and all futures.
const BROKERAGE_CAP_RATE: f64 = 0.0003;
/// Marker for segments where only the flat amount applies.
const FLAT_ONLY: f64 = 0.0;

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
                brokerage_flat: BROKERAGE,
                brokerage_rate: BROKERAGE_CAP_RATE,
                stt_rate: 0.00025, // 0.025% on the sell side
                stt_on: ChargedOn::Sell,
                exchange_txn_rate: 0.0000307, // NSE 0.00307%
                sebi_turnover_rate: SEBI,
                stamp_duty_rate: 0.00003, // 0.003% buy side
                gst_rate: GST,
            },
            Segment::EquityDelivery => CostSchedule {
                brokerage_flat: 0.0, // Zerodha: zero brokerage on delivery
                brokerage_rate: FLAT_ONLY,
                stt_rate: 0.001, // 0.1% on buy & sell
                stt_on: ChargedOn::Both,
                exchange_txn_rate: 0.0000307, // NSE 0.00307%
                sebi_turnover_rate: SEBI,
                stamp_duty_rate: 0.00015, // 0.015% buy side
                gst_rate: GST,
            },
            Segment::FuturesNfo => CostSchedule {
                brokerage_flat: BROKERAGE,
                brokerage_rate: BROKERAGE_CAP_RATE,
                stt_rate: 0.0005, // 0.05% sell side, effective 2026-04-01
                stt_on: ChargedOn::Sell,
                exchange_txn_rate: 0.0000183, // NSE 0.00183%
                sebi_turnover_rate: SEBI,
                stamp_duty_rate: 0.00002, // 0.002% buy side
                gst_rate: GST,
            },
            Segment::OptionsNfo => CostSchedule {
                brokerage_flat: BROKERAGE, // flat Rs 20, no percentage
                brokerage_rate: FLAT_ONLY,
                stt_rate: 0.0015, // 0.15% sell side on premium, eff. 2026-04-01
                stt_on: ChargedOn::Sell,
                exchange_txn_rate: 0.0003553, // NSE 0.03553% on premium
                sebi_turnover_rate: SEBI,
                stamp_duty_rate: 0.00003, // 0.003% buy side
                gst_rate: GST,
            },
            Segment::FuturesMcx => CostSchedule {
                brokerage_flat: BROKERAGE,
                brokerage_rate: BROKERAGE_CAP_RATE,
                stt_rate: 0.0001, // CTT 0.01% sell side (non-agri)
                stt_on: ChargedOn::Sell,
                exchange_txn_rate: 0.000021, // MCX 0.0021%
                sebi_turnover_rate: SEBI,
                stamp_duty_rate: 0.00002, // 0.002% buy side
                gst_rate: GST,
            },
            Segment::OptionsMcx => CostSchedule {
                brokerage_flat: BROKERAGE, // flat Rs 20, no percentage
                brokerage_rate: FLAT_ONLY,
                stt_rate: 0.0005, // CTT 0.05% sell side on premium
                stt_on: ChargedOn::Sell,
                exchange_txn_rate: 0.000418, // MCX 0.0418% on premium
                sebi_turnover_rate: SEBI,
                stamp_duty_rate: 0.00003, // 0.003% buy side
                gst_rate: GST,
            },
            Segment::FuturesCds => CostSchedule {
                brokerage_flat: BROKERAGE,
                brokerage_rate: BROKERAGE_CAP_RATE,
                stt_rate: 0.0, // no STT on currency
                stt_on: ChargedOn::Never,
                exchange_txn_rate: 0.0000035, // NSE 0.00035%
                sebi_turnover_rate: SEBI,
                stamp_duty_rate: 0.000001, // 0.0001% (Rs 10/crore) buy side
                gst_rate: GST,
            },
            Segment::OptionsCds => CostSchedule {
                brokerage_flat: BROKERAGE, // flat Rs 20, no percentage
                brokerage_rate: FLAT_ONLY,
                stt_rate: 0.0, // no STT on currency
                stt_on: ChargedOn::Never,
                exchange_txn_rate: 0.000311, // NSE 0.0311% on premium
                sebi_turnover_rate: SEBI,
                stamp_duty_rate: 0.000001, // 0.0001% (Rs 10/crore) buy side
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

    let mut fees =
        FeeBreakdown { brokerage: schedule.brokerage_for_order(value), ..Default::default() };

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
    fn currency_segments_have_no_stt() {
        for segment in [Segment::FuturesCds, Segment::OptionsCds] {
            let fees = calculate_side(segment, 1_000_000.0, Direction::Long, false);
            assert_eq!(fees.stt, 0.0, "{segment:?} should not levy STT");
        }
    }

    /// Commodity futures DO levy CTT (0.01% on the non-agri sell side). The
    /// schedule carried 0.0 for years -- "no STT for commodity futures" is
    /// true of the *securities* transaction tax but the commodities
    /// transaction tax fills the same line on the contract note.
    #[test]
    fn commodity_futures_levy_ctt_on_the_sell_side() {
        let sell = calculate_side(Segment::FuturesMcx, 1_000_000.0, Direction::Long, false);
        let buy = calculate_side(Segment::FuturesMcx, 1_000_000.0, Direction::Long, true);
        assert!((sell.stt - 0.0001 * 1_000_000.0).abs() < 1e-9);
        assert_eq!(buy.stt, 0.0);
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

    // ── Brokerage behaviour ─────────────────────────────────────────────────

    /// Zerodha charges ZERO brokerage on equity delivery. The old flat Rs 20
    /// charged money no broker collects -- 47 bps of phantom cost on a
    /// Rs 10,000 position, and it dominated STT below about Rs 21,000.
    #[test]
    fn equity_delivery_charges_no_brokerage() {
        for is_entry in [true, false] {
            let fees = calculate_side(Segment::EquityDelivery, 10_000.0, Direction::Long, is_entry);
            assert_eq!(fees.brokerage, 0.0);
        }
    }

    /// Intraday equity and futures brokerage is min(Rs 20, 0.03% of order
    /// value): below Rs 66,667 the percentage is cheaper.
    #[test]
    fn brokerage_cap_applies_below_the_crossover() {
        for segment in [Segment::EquityIntraday, Segment::FuturesNfo, Segment::FuturesMcx] {
            let small = calculate_side(segment, 50_000.0, Direction::Long, true);
            assert!(
                (small.brokerage - 15.0).abs() < 1e-9,
                "{segment:?}: 0.03% of 50k is Rs 15, under the Rs 20 cap"
            );
            let large = calculate_side(segment, 500_000.0, Direction::Long, true);
            assert_eq!(large.brokerage, 20.0, "{segment:?}: capped at Rs 20");
        }
    }

    /// Options brokerage is flat Rs 20 per order at any premium -- nb22/nb27
    /// class results depend on this staying flat, and Zerodha publishes it
    /// flat.
    #[test]
    fn options_brokerage_is_flat_at_any_size() {
        for value in [1_000.0, 50_000.0, 5_000_000.0] {
            let fees = calculate_side(Segment::OptionsNfo, value, Direction::Long, true);
            assert_eq!(fees.brokerage, 20.0);
        }
    }

    // ── Rate pinning ────────────────────────────────────────────────────────
    //
    // Every rate asserted against the schedule published at
    // https://zerodha.com/charges/, verified 2026-08-20. These exist because
    // the previous tests only checked schedules against each other, which let
    // the pre-2024 F&O STT rates survive two rounds of statutory increases
    // (2024-10-01 and 2026-04-01) undetected.

    #[test]
    fn pinned_equity_delivery_rates() {
        let s = Segment::EquityDelivery.schedule();
        assert_eq!(s.brokerage_flat, 0.0); // "Zero brokerage"
        assert_eq!(s.brokerage_rate, 0.0);
        assert_eq!(s.stt_rate, 0.001); // STT 0.1% on buy & sell
        assert_eq!(s.stt_on, ChargedOn::Both);
        assert_eq!(s.exchange_txn_rate, 0.0000307); // NSE 0.00307%
        assert_eq!(s.sebi_turnover_rate, 0.000001); // Rs 10/crore
        assert_eq!(s.stamp_duty_rate, 0.00015); // 0.015% buy side
        assert_eq!(s.gst_rate, 0.18);
    }

    #[test]
    fn pinned_equity_intraday_rates() {
        let s = Segment::EquityIntraday.schedule();
        assert_eq!(s.brokerage_flat, 20.0); // "0.03% or Rs 20, whichever lower"
        assert_eq!(s.brokerage_rate, 0.0003);
        assert_eq!(s.stt_rate, 0.00025); // STT 0.025% sell side
        assert_eq!(s.stt_on, ChargedOn::Sell);
        assert_eq!(s.exchange_txn_rate, 0.0000307); // NSE 0.00307%
        assert_eq!(s.stamp_duty_rate, 0.00003); // 0.003% buy side
    }

    #[test]
    fn pinned_nfo_futures_rates() {
        let s = Segment::FuturesNfo.schedule();
        assert_eq!(s.brokerage_flat, 20.0);
        assert_eq!(s.brokerage_rate, 0.0003);
        // STT on sale of futures in securities: 0.05%, effective 2026-04-01
        // (Budget 2026-27; was 0.02% from 2024-10-01, 0.0125% before that).
        assert_eq!(s.stt_rate, 0.0005);
        assert_eq!(s.stt_on, ChargedOn::Sell);
        assert_eq!(s.exchange_txn_rate, 0.0000183); // NSE 0.00183%
        assert_eq!(s.stamp_duty_rate, 0.00002); // 0.002% buy side
    }

    #[test]
    fn pinned_nfo_options_rates() {
        let s = Segment::OptionsNfo.schedule();
        assert_eq!(s.brokerage_flat, 20.0); // flat Rs 20 per executed order
        assert_eq!(s.brokerage_rate, 0.0);
        // STT on sale of an option: 0.15% of premium, effective 2026-04-01
        // (Budget 2026-27; was 0.1% from 2024-10-01, 0.0625% before that).
        assert_eq!(s.stt_rate, 0.0015);
        assert_eq!(s.stt_on, ChargedOn::Sell);
        assert_eq!(s.exchange_txn_rate, 0.0003553); // NSE 0.03553% on premium
        assert_eq!(s.stamp_duty_rate, 0.00003); // 0.003% buy side
        assert!(Segment::OptionsNfo.charges_on_premium());
    }

    #[test]
    fn pinned_mcx_rates() {
        let fut = Segment::FuturesMcx.schedule();
        assert_eq!(fut.brokerage_rate, 0.0003);
        assert_eq!(fut.stt_rate, 0.0001); // CTT 0.01% sell side, non-agri
        assert_eq!(fut.stt_on, ChargedOn::Sell);
        assert_eq!(fut.exchange_txn_rate, 0.000021); // MCX 0.0021%
        assert_eq!(fut.stamp_duty_rate, 0.00002);

        let opt = Segment::OptionsMcx.schedule();
        assert_eq!(opt.brokerage_rate, 0.0);
        assert_eq!(opt.stt_rate, 0.0005); // CTT 0.05% sell side on premium
        assert_eq!(opt.exchange_txn_rate, 0.000418); // MCX 0.0418% on premium
        assert_eq!(opt.stamp_duty_rate, 0.00003);
    }

    #[test]
    fn pinned_cds_rates() {
        let fut = Segment::FuturesCds.schedule();
        assert_eq!(fut.brokerage_rate, 0.0003);
        assert_eq!(fut.stt_rate, 0.0); // no STT on currency
        assert_eq!(fut.exchange_txn_rate, 0.0000035); // NSE 0.00035%
        assert_eq!(fut.stamp_duty_rate, 0.000001); // 0.0001% (Rs 10/crore)

        let opt = Segment::OptionsCds.schedule();
        assert_eq!(opt.brokerage_rate, 0.0);
        assert_eq!(opt.stt_rate, 0.0);
        assert_eq!(opt.exchange_txn_rate, 0.000311); // NSE 0.0311% on premium
        assert_eq!(opt.stamp_duty_rate, 0.000001);
    }
}
