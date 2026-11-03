//! Itemised Indian transaction costs for the engine's built-in segments.
//!
//! The engine charges fees per side (once on entry, once on exit), so every
//! rate is applied to one leg. Regulatory schedules are usually quoted per
//! round trip -- STT "on sell side", stamp duty "on buy side" -- which is why
//! each component states which side it lands on.
//!
//! The rates are not kept here: they are the dated statutory exchange rows
//! of [`crate::execution::costs`] (`exchange_rates`), the same rows a user
//! loads or overrides as a [`FeeSchedule`](crate::execution::costs::FeeSchedule),
//! at the latest date in force. [`Segment`] is the enum the engine's
//! `fee_segment` string selects, and [`calculate_side`] charges one side
//! through the single computation in
//! [`FeeRates::charge_fill`](crate::execution::costs::FeeRates::charge_fill),
//! so the equity curve and the reported breakdown are the same money.
//!
//! Every rate is pinned field by field in this module's tests. The rows carry
//! no brokerage and no depository charge: a broker's own charges come from its
//! rate card. BSE equity and BFO derivatives have rows of their own in the
//! dated schedule; [`Segment`] names the NSE-family segments only and a run
//! priced by venue and kind reaches the others. The tax on options that are
//! *exercised* is the settlement row of the exchange schedule, charged by the
//! settlement path on the intrinsic value (buyer side only).

use crate::core::types::Direction;
use crate::execution::costs::{exchange_rates, FeeRates, FeeSegment};

pub use crate::execution::costs::FeeBreakdown;

/// Market segment and instrument type, selecting a built-in cost row.
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

/// Depository participant charge on equity delivery sells, with GST:
/// 13.00 + 18% = 15.34.
///
/// This is **per ISIN per day with any sell**, not per order -- selling one
/// scrip across five orders in a day incurs it once, selling five scrips
/// incurs it five times. That unit is why it is a per-instrument-day fee
/// (`DayFee`) on the delivery row and not part of the per-fill rates:
/// folding it into them would either overcount multi-order days or
/// undercount multi-scrip days. Consumers (the rebalance simulator, and any
/// small-book refusal arithmetic built on it) count distinct ISINs with net
/// sells per day and multiply.
///
/// This flat charge, not the percentage rates, is what dominates rebalancing
/// costs on small delivery books -- at a 3,000 sell it is ~0.5% before STT.
pub const DP_SELL_CHARGE_PER_ISIN_PER_DAY: f64 = 15.34;

impl Segment {
    /// The venue and row segment of the built-in schedule this selects.
    pub fn venue_segment(&self) -> (&'static str, FeeSegment) {
        match self {
            Segment::EquityIntraday => ("NSE", FeeSegment::Intraday),
            Segment::EquityDelivery => ("NSE", FeeSegment::Delivery),
            Segment::FuturesNfo => ("NFO", FeeSegment::Futures),
            Segment::OptionsNfo => ("NFO", FeeSegment::Options),
            Segment::FuturesMcx => ("MCX", FeeSegment::Futures),
            Segment::OptionsMcx => ("MCX", FeeSegment::Options),
            Segment::FuturesCds => ("CDS", FeeSegment::Futures),
            Segment::OptionsCds => ("CDS", FeeSegment::Options),
        }
    }

    /// Rates for this segment.
    pub fn rates(&self) -> FeeRates {
        let (venue, segment) = self.venue_segment();
        exchange_rates(venue, segment).expect("every Segment has a built-in row")
    }

    /// Whether STT and exchange charges apply to option premium rather than
    /// contract notional.
    #[inline]
    pub fn charges_on_premium(&self) -> bool {
        matches!(self, Segment::OptionsNfo | Segment::OptionsMcx | Segment::OptionsCds)
    }

    /// Parse a segment name, as it appears on an instrument record.
    ///
    /// `intraday` distinguishes equity intraday from delivery; it is ignored
    /// for derivative segments, which are always intraday-rated.
    pub fn parse(segment: &str, instrument_type: Option<&str>, intraday: bool) -> Option<Self> {
        let (venue, seg) = parse_venue_segment(segment, instrument_type, intraday)?;
        Some(match (venue.as_str(), seg) {
            ("NSE" | "BSE", FeeSegment::Intraday) => Segment::EquityIntraday,
            ("NSE" | "BSE", _) => Segment::EquityDelivery,
            ("NFO" | "BFO", FeeSegment::Options) => Segment::OptionsNfo,
            ("NFO" | "BFO", _) => Segment::FuturesNfo,
            ("MCX", FeeSegment::Options) => Segment::OptionsMcx,
            ("MCX", _) => Segment::FuturesMcx,
            (_, FeeSegment::Options) => Segment::OptionsCds,
            _ => Segment::FuturesCds,
        })
    }
}

/// The venue (upper-cased) and row segment a `fee_segment` name selects,
/// without choosing a rate: `("BFO", Options)` for `"bfo"` and `"OPT"`.
/// `None` for a venue the built-in schedule does not price.
pub fn parse_venue_segment(
    segment: &str,
    instrument_type: Option<&str>,
    intraday: bool,
) -> Option<(String, FeeSegment)> {
    let venue = segment.to_ascii_uppercase();
    let ty = instrument_type.map(|t| t.to_ascii_uppercase());
    let is_option = matches!(ty.as_deref(), Some("OPT") | Some("CE") | Some("PE"));
    let seg = match venue.as_str() {
        "NSE" | "BSE" => {
            if intraday {
                FeeSegment::Intraday
            } else {
                FeeSegment::Delivery
            }
        }
        "NFO" | "BFO" | "MCX" | "CDS" | "BCD" => {
            if is_option {
                FeeSegment::Options
            } else {
                FeeSegment::Futures
            }
        }
        _ => return None,
    };
    Some((venue, seg))
}

/// Itemised cost for **one side** of a trade.
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
    // A long entry buys and a long exit sells; a short is the reverse.
    let is_buy =
        matches!((direction, is_entry), (Direction::Long, true) | (Direction::Short, false));
    segment.rates().charge(is_buy, value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::costs::TaxKind;

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
        let expected = 0.18 * (fees.brokerage + fees.exchange_txn + fees.ipft + fees.sebi_fee);
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
            + fees.ipft
            + fees.dp_charge
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

    /// The built-in rows are what the exchanges and the tax authority levy:
    /// no brokerage at any size, and no depository charge. Brokerage comes
    /// from a broker's rate card.
    #[test]
    fn the_built_in_rows_charge_no_brokerage_or_depository_fee() {
        let all = [
            Segment::EquityIntraday,
            Segment::EquityDelivery,
            Segment::FuturesNfo,
            Segment::OptionsNfo,
            Segment::FuturesMcx,
            Segment::OptionsMcx,
            Segment::FuturesCds,
            Segment::OptionsCds,
        ];
        for segment in all {
            for value in [1_000.0, 50_000.0, 5_000_000.0] {
                for is_entry in [true, false] {
                    let fees = calculate_side(segment, value, Direction::Long, is_entry);
                    assert_eq!((fees.brokerage, fees.dp_charge), (0.0, 0.0), "{segment:?}");
                }
            }
            assert!(segment.rates().day_fee.is_none());
        }
    }

    // ── Rate pinning ────────────────────────────────────────────────────────
    //
    // Every rate asserted against the published exchange, clearing-corporation
    // and regulator schedules, checked 2026-08-20. These exist because the
    // previous tests only checked schedules against each other, which let the
    // pre-2024 F&O STT rates survive two rounds of statutory increases
    // (2024-10-01 and 2026-04-01) undetected.

    fn capped(r: &FeeRates) -> (f64, f64, Option<f64>) {
        (r.brokerage_flat, r.brokerage_rate, r.brokerage_cap)
    }

    #[test]
    fn pinned_equity_delivery_rates() {
        let s = Segment::EquityDelivery.rates();
        assert_eq!(capped(&s), (0.0, 0.0, None)); // no brokerage in a preset
        assert_eq!((s.tax_buy_rate, s.tax_sell_rate), (0.001, 0.001)); // STT 0.1% both
        assert_eq!(s.exchange_rate, 306.99 / 1.0e7); // NSE/FA/73061: 306.99 per crore
        assert_eq!(s.investor_fund_rate, 0.01 / 1.0e7); // IPFT 0.01 per crore
        assert_eq!(s.regulatory_rate, 0.000001); // 10 per crore
        assert_eq!(s.stamp_buy_rate, 0.00015); // 0.015% buy side
        assert_eq!(s.gst_rate, 0.18);
        assert!(s.day_fee.is_none(), "the depository charge is the broker's, on its rate card");
    }

    #[test]
    fn pinned_equity_intraday_rates() {
        let s = Segment::EquityIntraday.rates();
        assert_eq!(capped(&s), (0.0, 0.0, None));
        assert_eq!((s.tax_buy_rate, s.tax_sell_rate), (0.0, 0.00025)); // STT 0.025% sell
        assert_eq!(s.exchange_rate, 306.99 / 1.0e7); // NSE/FA/73061: 306.99 per crore
        assert_eq!(s.investor_fund_rate, 0.01 / 1.0e7); // IPFT 0.01 per crore
        assert_eq!(s.stamp_buy_rate, 0.00003); // 0.003% buy side
    }

    #[test]
    fn pinned_nfo_futures_rates() {
        let s = Segment::FuturesNfo.rates();
        assert_eq!(capped(&s), (0.0, 0.0, None));
        // STT on sale of futures in securities: 0.05%, effective 2026-04-01
        // (was 0.02% from 2024-10-01, 0.0125% before that).
        assert_eq!((s.tax_buy_rate, s.tax_sell_rate), (0.0, 0.0005));
        assert_eq!(s.exchange_rate, 182.99 / 1.0e7); // futures 182.99 per crore
        assert_eq!(s.stamp_buy_rate, 0.00002); // 0.002% buy side
    }

    #[test]
    fn pinned_nfo_options_rates() {
        let s = Segment::OptionsNfo.rates();
        assert_eq!(capped(&s), (0.0, 0.0, None));
        // STT on sale of an option: 0.15% of premium, effective 2026-04-01
        // (was 0.1% from 2024-10-01).
        assert_eq!((s.tax_buy_rate, s.tax_sell_rate), (0.0, 0.0015));
        assert_eq!(s.exchange_rate, 3_552.0 / 1.0e7); // options 3,552 per crore of premium
        assert_eq!(s.stamp_buy_rate, 0.00003); // 0.003% buy side
        assert!(Segment::OptionsNfo.charges_on_premium());
    }

    #[test]
    fn pinned_mcx_rates() {
        let fut = Segment::FuturesMcx.rates();
        assert_eq!(fut.brokerage_rate, 0.0);
        assert_eq!(fut.tax_kind, TaxKind::Ctt);
        assert_eq!(fut.tax_sell_rate, 0.0001); // CTT 0.01% sell side, non-agri
        assert_eq!(fut.exchange_rate, 210.0 / 1.0e7); // Rs 2.10 per lakh (unverified)
        assert_eq!(fut.stamp_buy_rate, 0.00002);

        let opt = Segment::OptionsMcx.rates();
        assert_eq!(opt.brokerage_rate, 0.0);
        assert_eq!(opt.tax_sell_rate, 0.0005); // CTT 0.05% sell side on premium
        assert_eq!(opt.exchange_rate, 4_180.0 / 1.0e7); // Rs 41.80 per lakh (unverified)
        assert_eq!(opt.stamp_buy_rate, 0.00003);
    }

    #[test]
    fn pinned_cds_rates() {
        let fut = Segment::FuturesCds.rates();
        assert_eq!(fut.brokerage_rate, 0.0);
        assert_eq!(fut.tax_kind, TaxKind::None); // no STT on currency
        assert_eq!(fut.exchange_rate, 35.0 / 1.0e7); // 0.00035% (unverified)
        assert_eq!(fut.stamp_buy_rate, 0.000001); // 0.0001% (10 per crore)

        let opt = Segment::OptionsCds.rates();
        assert_eq!(opt.brokerage_rate, 0.0);
        assert_eq!(opt.tax_kind, TaxKind::None);
        assert_eq!(opt.exchange_rate, 3_110.0 / 1.0e7); // 0.0311% on premium (unverified)
        assert_eq!(opt.stamp_buy_rate, 0.000001);
    }

    /// One hand-computed side of each kind, from the per-crore figures: the
    /// itemised parts are separate (exchange charge, IPFT, SEBI fee, tax, stamp
    /// duty) and GST is 18% of exchange + IPFT + SEBI.
    #[test]
    fn a_nfo_option_sale_is_itemised_from_the_per_crore_figures() {
        // 150,000 of premium sold, after 2026-04-01: STT 0.15%, exchange 3,552/crore,
        // IPFT 0.01/crore, SEBI 10/crore, no stamp duty on a sale.
        let b = calculate_side(Segment::OptionsNfo, 150_000.0, Direction::Long, false);
        let crore = 1.0e7;
        assert!((b.stt - 150_000.0 * 15_000.0 / crore).abs() < 1e-9);
        assert!((b.exchange_txn - 150_000.0 * 3_552.0 / crore).abs() < 1e-9);
        assert!((b.ipft - 150_000.0 * 0.01 / crore).abs() < 1e-12);
        assert!((b.sebi_fee - 150_000.0 * 10.0 / crore).abs() < 1e-9);
        assert_eq!(b.stamp_duty, 0.0);
        let gst = 0.18 * (b.exchange_txn + b.ipft + b.sebi_fee);
        assert!((b.gst - gst).abs() < 1e-9, "GST is on exchange + IPFT + SEBI, never on the tax");
        assert!((b.total() - (b.stt + b.exchange_txn + b.ipft + b.sebi_fee + b.gst)).abs() < 1e-9);
    }
}
