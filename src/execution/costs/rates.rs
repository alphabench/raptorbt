//! The rates that price one fill, and the itemised charge they produce.
//!
//! [`FeeRates`] is plain numbers (`Copy`, no strings), so the kernel can hold
//! one and charge it per fill without allocating. Which rates apply to which
//! venue and segment is the job of [`super::schedule`].
//!
//! Every rate is a fraction of turnover (0.0003 is 0.03%); money is in the
//! currency's major unit. `value` is what the rates apply to: price times
//! quantity times multiplier, which for an option is the premium traded.

use super::rounding::{paise, whole_rupees, Round};
use serde::{Deserialize, Serialize};

/// What the transaction tax is called on the contract note.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TaxKind {
    /// Securities transaction tax.
    #[default]
    Stt,
    /// Commodities transaction tax.
    Ctt,
    /// No transaction tax.
    None,
}

impl TaxKind {
    pub fn as_str(self) -> &'static str {
        match self {
            TaxKind::Stt => "STT",
            TaxKind::Ctt => "CTT",
            TaxKind::None => "NONE",
        }
    }
}

/// How the components of a charge are rounded. See
/// `docs/quantitative-correctness/numerics.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Rounding {
    /// Not rounded: the exact `f64` products. The built-in exchange presets.
    #[default]
    None,
    /// Each component to the paisa, half up, once computed; GST on the
    /// rounded parts.
    PaiseEach,
    /// As a contract note: stamp duty in whole rupees (half up), GST on the
    /// unrounded brokerage, exchange and regulatory charges, each part shown
    /// to the paisa half-even, and the total the rounded sum of the
    /// unrounded parts (which can differ by a paisa from the sum of the
    /// shown parts).
    ContractNote,
}

impl Rounding {
    pub fn as_str(self) -> &'static str {
        match self {
            Rounding::None => "none",
            Rounding::PaiseEach => "paise_each",
            Rounding::ContractNote => "contract_note",
        }
    }
}

/// What the flat brokerage and its cap apply to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum BrokeragePer {
    /// Each fill is its own order.
    #[default]
    Fill,
    /// The order: the flat amount is charged on the first fill and the cap
    /// holds over the order's fills together. Needs the order's turnover so
    /// far ([`FeeRates::charge_fill`]).
    Order,
}

impl BrokeragePer {
    pub fn as_str(self) -> &'static str {
        match self {
            BrokeragePer::Fill => "fill",
            BrokeragePer::Order => "order",
        }
    }
}

/// Which components the GST (or VAT) is levied on. Declared per row because
/// it differs by regime: Indian GST is on brokerage and exchange-side
/// charges, never on the transaction tax or stamp duty (themselves taxes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GstBase {
    pub brokerage: bool,
    pub exchange: bool,
    pub regulatory: bool,
    pub investor_fund: bool,
    pub transaction_tax: bool,
    pub stamp_duty: bool,
}

impl GstBase {
    /// Brokerage, exchange and regulatory charges.
    pub const SERVICES: GstBase = GstBase {
        brokerage: true,
        exchange: true,
        regulatory: true,
        investor_fund: false,
        transaction_tax: false,
        stamp_duty: false,
    };
}

impl Default for GstBase {
    fn default() -> Self {
        GstBase::SERVICES
    }
}

/// Which side a per-instrument-per-day fee falls on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DaySide {
    Buy,
    Sell,
}

/// A flat fee owed once per instrument per day on one side (the depository
/// charge on a delivery sell). The caller says whether the day owes it: a
/// single fill cannot know whether an earlier fill that day paid it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DayFee {
    pub amount: f64,
    pub side: DaySide,
    /// Whether the GST rate applies to the amount.
    pub gst: bool,
}

/// The rates of one row.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FeeRates {
    /// Flat brokerage per order (or per fill).
    pub brokerage_flat: f64,
    /// Percentage brokerage, on the order's value.
    pub brokerage_rate: f64,
    /// Brokerage never exceeds this per order ("0.03% or 20, whichever is
    /// lower" is a rate of 0.0003 with a cap of 20).
    pub brokerage_cap: Option<f64>,
    /// Brokerage is never below this per order.
    pub brokerage_min: f64,
    pub brokerage_per: BrokeragePer,
    pub tax_kind: TaxKind,
    /// Transaction tax rate on a buy.
    pub tax_buy_rate: f64,
    /// Transaction tax rate on a sell.
    pub tax_sell_rate: f64,
    /// Exchange transaction charge.
    pub exchange_rate: f64,
    /// Regulatory turnover fee.
    pub regulatory_rate: f64,
    /// Investor protection fund charge.
    pub investor_fund_rate: f64,
    /// Stamp duty, on the buy side.
    pub stamp_buy_rate: f64,
    pub gst_rate: f64,
    pub gst_base: GstBase,
    pub day_fee: Option<DayFee>,
    pub rounding: Rounding,
}

impl FeeRates {
    /// Rates that charge nothing.
    pub const ZERO: FeeRates = FeeRates {
        brokerage_flat: 0.0,
        brokerage_rate: 0.0,
        brokerage_cap: None,
        brokerage_min: 0.0,
        brokerage_per: BrokeragePer::Fill,
        tax_kind: TaxKind::None,
        tax_buy_rate: 0.0,
        tax_sell_rate: 0.0,
        exchange_rate: 0.0,
        regulatory_rate: 0.0,
        investor_fund_rate: 0.0,
        stamp_buy_rate: 0.0,
        gst_rate: 0.0,
        gst_base: GstBase::SERVICES,
        day_fee: None,
        rounding: Rounding::None,
    };

    /// Unrounded brokerage of an order whose turnover is `t`.
    fn brokerage_at(&self, t: f64, round_pct: bool) -> f64 {
        let pct = if self.brokerage_rate > 0.0 {
            let p = self.brokerage_rate * t;
            if round_pct {
                paise(p, Round::HalfUp)
            } else {
                p
            }
        } else {
            0.0
        };
        let mut b = self.brokerage_flat + pct;
        if let Some(cap) = self.brokerage_cap {
            b = b.min(cap);
        }
        b.max(self.brokerage_min)
    }

    /// Brokerage of one fill of `value` after `order_before` of the same
    /// order already traded.
    fn brokerage(&self, value: f64, order_before: f64, round_pct: bool) -> f64 {
        match self.brokerage_per {
            // A contract note prices no brokerage on a zero-value fill.
            BrokeragePer::Fill if self.rounding == Rounding::ContractNote && value == 0.0 => 0.0,
            BrokeragePer::Fill => self.brokerage_at(value, round_pct),
            BrokeragePer::Order => {
                let upto = |t: f64| {
                    if t == 0.0 {
                        0.0
                    } else {
                        self.brokerage_at(t, round_pct)
                    }
                };
                upto(order_before + value) - upto(order_before)
            }
        }
    }

    /// The charge on one side of a trade, each fill its own order.
    #[inline]
    pub fn charge(&self, is_buy: bool, value: f64) -> FeeBreakdown {
        self.charge_fill(is_buy, value.abs(), 0.0, false)
    }

    /// The charge on one fill. `order_before` is what earlier fills of the
    /// same order already traded (it matters only when brokerage is per
    /// order); `day_fee_owed` says the instrument's per-day fee falls due on
    /// this fill.
    pub fn charge_fill(
        &self,
        is_buy: bool,
        value: f64,
        order_before: f64,
        day_fee_owed: bool,
    ) -> FeeBreakdown {
        self.charge_inner(is_buy, value, order_before, day_fee_owed, None)
    }

    /// The charge on one fill with the brokerage given: what was booked, not what the row would
    /// charge now. The other components are the row's, and GST is taken on the given brokerage.
    pub fn charge_keeping_brokerage(
        &self,
        is_buy: bool,
        value: f64,
        brokerage: f64,
    ) -> FeeBreakdown {
        self.charge_inner(is_buy, value.abs(), 0.0, false, Some(brokerage))
    }

    fn charge_inner(
        &self,
        is_buy: bool,
        value: f64,
        order_before: f64,
        day_fee_owed: bool,
        booked_brokerage: Option<f64>,
    ) -> FeeBreakdown {
        let tax_rate = if is_buy { self.tax_buy_rate } else { self.tax_sell_rate };
        let tax_rate = if self.tax_kind == TaxKind::None { 0.0 } else { tax_rate };
        let round_each = self.rounding == Rounding::PaiseEach;
        let up = |x: f64| if round_each { paise(x, Round::HalfUp) } else { x };

        let brokerage =
            booked_brokerage.unwrap_or_else(|| self.brokerage(value, order_before, round_each));
        let tax = if tax_rate > 0.0 { up(tax_rate * value) } else { 0.0 };
        let exchange = up(self.exchange_rate * value);
        let regulatory = up(self.regulatory_rate * value);
        let investor_fund =
            if self.investor_fund_rate > 0.0 { up(self.investor_fund_rate * value) } else { 0.0 };
        let stamp = if is_buy {
            let s = self.stamp_buy_rate * value;
            match self.rounding {
                Rounding::None => s,
                Rounding::PaiseEach => paise(s, Round::HalfUp),
                Rounding::ContractNote => whole_rupees(s),
            }
        } else {
            0.0
        };

        let g = self.gst_base;
        let mut base = 0.0;
        for (on, x) in [
            (g.brokerage, brokerage),
            (g.exchange, exchange),
            (g.regulatory, regulatory),
            (g.investor_fund, investor_fund),
            (g.transaction_tax, tax),
            (g.stamp_duty, stamp),
        ] {
            if on {
                base += x;
            }
        }
        let gst = up(self.gst_rate * base);

        let mut out = match self.rounding {
            Rounding::None => FeeBreakdown {
                brokerage,
                stt: tax,
                exchange_txn: exchange,
                sebi_fee: regulatory,
                stamp_duty: stamp,
                gst,
                ipft: investor_fund,
                dp_charge: 0.0,
                exact_total: None,
            },
            Rounding::PaiseEach => {
                let r = |x: f64| paise(x, Round::HalfUp);
                let mut b = FeeBreakdown {
                    brokerage: r(brokerage),
                    stt: r(tax),
                    exchange_txn: r(exchange),
                    sebi_fee: r(regulatory),
                    stamp_duty: r(stamp),
                    gst: r(gst),
                    ipft: r(investor_fund),
                    dp_charge: 0.0,
                    exact_total: None,
                };
                b.exact_total = Some(r(b.sum()));
                b
            }
            Rounding::ContractNote => {
                let r = |x: f64| paise(x, Round::HalfEven);
                let total = brokerage + tax + exchange + regulatory + investor_fund + stamp + gst;
                FeeBreakdown {
                    brokerage: r(brokerage),
                    stt: r(tax),
                    exchange_txn: r(exchange),
                    sebi_fee: r(regulatory),
                    stamp_duty: r(stamp),
                    gst: r(gst),
                    ipft: r(investor_fund),
                    dp_charge: 0.0,
                    exact_total: Some(r(total)),
                }
            }
        };
        if let (Some(day), true) = (self.day_fee, day_fee_owed) {
            if is_buy == (day.side == DaySide::Buy) {
                out.apply_day_fee(self, day);
            }
        }
        out
    }
}

/// Itemised costs for one side of a trade.
///
/// `stt` holds the transaction tax whatever its name on the note (the row's
/// [`TaxKind`] says which). `exact_total` is set by the rounded modes: the
/// total as the note prints it, which can differ by a paisa from the sum of
/// the shown parts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct FeeBreakdown {
    pub brokerage: f64,
    pub stt: f64,
    pub exchange_txn: f64,
    pub sebi_fee: f64,
    pub stamp_duty: f64,
    pub gst: f64,
    /// Investor protection fund charge.
    pub ipft: f64,
    /// The per-instrument-per-day fee, when it fell due on this fill.
    pub dp_charge: f64,
    pub exact_total: Option<f64>,
}

impl FeeBreakdown {
    fn sum(&self) -> f64 {
        self.brokerage
            + self.stt
            + self.exchange_txn
            + self.sebi_fee
            + self.stamp_duty
            + self.gst
            + self.ipft
            + self.dp_charge
    }

    /// Add the per-instrument-per-day fee (and its GST) to a charge, rounded
    /// as the row rounds.
    fn apply_day_fee(&mut self, rates: &FeeRates, day: DayFee) {
        let gst = if day.gst { rates.gst_rate * day.amount } else { 0.0 };
        match rates.rounding {
            Rounding::None => {
                self.dp_charge += day.amount;
                self.gst += gst;
            }
            Rounding::PaiseEach => {
                let r = |x: f64| paise(x, Round::HalfUp);
                self.dp_charge += r(day.amount);
                self.gst = r(self.gst + r(gst));
                self.exact_total = Some(r(self.sum()));
            }
            Rounding::ContractNote => {
                let r = |x: f64| paise(x, Round::HalfEven);
                let (dp, gst) = (r(day.amount), r(gst));
                self.dp_charge += dp;
                self.gst = r(self.gst + gst);
                self.exact_total = self.exact_total.map(|t| r(r(t + dp) + gst));
            }
        }
    }

    /// What is charged: `exact_total` when the row rounds, else the sum of
    /// the components. This is the number that reaches the equity curve.
    #[inline]
    pub fn total(&self) -> f64 {
        self.exact_total.unwrap_or_else(|| self.sum())
    }

    /// Accumulate another breakdown, for aggregating across trades.
    pub fn add(&mut self, other: &FeeBreakdown) {
        let total = self.total() + other.total();
        let exact = self.exact_total.is_some() || other.exact_total.is_some();
        self.brokerage += other.brokerage;
        self.stt += other.stt;
        self.exchange_txn += other.exchange_txn;
        self.sebi_fee += other.sebi_fee;
        self.stamp_duty += other.stamp_duty;
        self.gst += other.gst;
        self.ipft += other.ipft;
        self.dp_charge += other.dp_charge;
        self.exact_total = if exact { Some(total) } else { None };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn note() -> FeeRates {
        FeeRates {
            brokerage_rate: 0.0003,
            brokerage_cap: Some(20.0),
            brokerage_per: BrokeragePer::Order,
            tax_kind: TaxKind::Stt,
            tax_sell_rate: 0.00025,
            exchange_rate: 0.0000307,
            regulatory_rate: 0.000001,
            stamp_buy_rate: 0.00003,
            gst_rate: 0.18,
            rounding: Rounding::ContractNote,
            ..FeeRates::ZERO
        }
    }

    #[test]
    fn a_contract_note_total_is_the_rounded_sum_of_the_unrounded_parts() {
        let b = note().charge_fill(false, 15_005.5, 0.0, false);
        assert_eq!(b.brokerage, 4.5);
        assert_eq!(b.stt, 3.75);
        assert_eq!(b.stamp_duty, 0.0);
        assert_eq!(b.exact_total, Some(9.62));
        assert_eq!(b.total(), 9.62);
    }

    #[test]
    fn stamp_duty_is_whole_rupees_on_a_contract_note_and_only_on_a_buy() {
        let buy = note().charge_fill(true, 10_000_000.0, 0.0, false);
        assert_eq!(buy.stamp_duty, 300.0);
        let sell = note().charge_fill(false, 10_000_000.0, 0.0, false);
        assert_eq!(sell.stamp_duty, 0.0);
    }

    #[test]
    fn the_brokerage_cap_holds_over_the_fills_of_one_order() {
        let first = note().charge_fill(true, 50_000.0, 0.0, false);
        let second = note().charge_fill(true, 50_000.0, 50_000.0, false);
        assert_eq!(first.brokerage, 15.0);
        assert_eq!(second.brokerage, 5.0);
    }

    #[test]
    fn the_day_fee_and_its_gst_follow_the_row_rounding() {
        let rates = FeeRates {
            day_fee: Some(DayFee { amount: 13.0, side: DaySide::Sell, gst: true }),
            ..note()
        };
        let plain = rates.charge_fill(false, 1000.0, 0.0, false);
        let owed = rates.charge_fill(false, 1000.0, 0.0, true);
        assert_eq!(plain.dp_charge, 0.0);
        assert_eq!(owed.dp_charge, 13.0);
        assert_eq!(owed.gst, paise(plain.gst + 2.34, Round::HalfEven));
        assert!(owed.total() > plain.total() + 15.0);
        // a buy never owes a sell-side fee
        assert_eq!(rates.charge_fill(true, 1000.0, 0.0, true).dp_charge, 0.0);
    }

    #[test]
    fn an_accumulated_breakdown_keeps_the_total_it_was_charged() {
        let mut a = note().charge_fill(true, 15_005.5, 0.0, false);
        let b = note().charge_fill(false, 15_005.5, 0.0, false);
        let want = a.total() + b.total();
        a.add(&b);
        assert_eq!(a.total(), want);
    }

    #[test]
    fn gst_is_levied_on_the_declared_base() {
        let mut r = note();
        r.rounding = Rounding::None;
        r.gst_base = GstBase { transaction_tax: true, ..GstBase::SERVICES };
        let b = r.charge_fill(false, 100_000.0, 0.0, false);
        let want = 0.18 * (b.brokerage + b.exchange_txn + b.sebi_fee + b.stt);
        assert!((b.gst - want).abs() < 1e-12);
    }
}
