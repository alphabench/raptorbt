//! Fee rows by venue, segment and underlying, and the schedule that picks
//! one for a fill.

use super::rates::{
    BrokeragePer, DayFee, DaySide, FeeBreakdown, FeeRates, GstBase, Rounding, TaxKind,
};
use crate::core::error::{RaptorError, Result};

/// The segment of a venue a row prices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FeeSegment {
    /// Equity held overnight (delivery).
    Delivery,
    /// Equity squared off the same day.
    Intraday,
    Futures,
    Options,
}

impl FeeSegment {
    pub fn as_str(self) -> &'static str {
        match self {
            FeeSegment::Delivery => "DELIVERY",
            FeeSegment::Intraday => "INTRADAY",
            FeeSegment::Futures => "FUTURES",
            FeeSegment::Options => "OPTIONS",
        }
    }

    pub fn parse(text: &str) -> Result<Self> {
        Ok(match text {
            "DELIVERY" => FeeSegment::Delivery,
            "INTRADAY" => FeeSegment::Intraday,
            "FUTURES" => FeeSegment::Futures,
            "OPTIONS" => FeeSegment::Options,
            other => {
                return Err(RaptorError::invalid_parameter(format!(
                    "unknown fee segment {other:?}: DELIVERY, INTRADAY, FUTURES or OPTIONS"
                )))
            }
        })
    }

    /// Whether the rates apply to option premium rather than notional.
    pub fn on_premium(self) -> bool {
        self == FeeSegment::Options
    }
}

/// One row: the rates for a venue and segment, optionally for one
/// underlying, from a date.
#[derive(Debug, Clone, PartialEq)]
pub struct FeeRow {
    pub venue: String,
    pub segment: FeeSegment,
    /// A row for one underlying only (for example an agricultural
    /// commodity), by symbol; it beats the venue's general row.
    pub underlying: Option<String>,
    /// A row that prices a contract's settlement at expiry, never a trade.
    pub settlement: bool,
    /// UTC nanoseconds. A fill before every row's date is priced on the
    /// earliest row.
    pub effective_from: i64,
    pub rates: FeeRates,
}

impl FeeRow {
    pub fn label(&self) -> String {
        let mut s = format!("{}:{}", self.venue, self.segment.as_str());
        if let Some(u) = &self.underlying {
            s.push(':');
            s.push_str(u);
        }
        if self.settlement {
            s.push_str(":SETTLEMENT");
        }
        s
    }
}

/// A named set of rows.
#[derive(Debug, Clone, PartialEq)]
pub struct FeeSchedule {
    pub name: String,
    /// Where the rates come from, with their date.
    pub source: String,
    pub rows: Vec<FeeRow>,
}

impl FeeSchedule {
    pub fn new(name: impl Into<String>, source: impl Into<String>, rows: Vec<FeeRow>) -> Self {
        FeeSchedule { name: name.into(), source: source.into(), rows }
    }

    /// One schedule from several, their rows in order.
    pub fn merge(parts: &[FeeSchedule]) -> Result<Self> {
        if parts.is_empty() {
            return Err(RaptorError::invalid_parameter("merge needs at least one schedule"));
        }
        let name = parts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join("+");
        let source = parts.iter().map(|p| p.source.as_str()).collect::<Vec<_>>().join("; ");
        Ok(FeeSchedule::new(name, source, parts.iter().flat_map(|p| p.rows.clone()).collect()))
    }

    /// The row that prices a fill on `venue` in `segment` at `at` (UTC ns).
    ///
    /// The row for the instrument's own underlying beats the general one;
    /// among rows in force at `at` the latest wins; a fill before every row
    /// takes the earliest. No row is an error, never a free fill.
    pub fn row_for(
        &self,
        venue: &str,
        segment: FeeSegment,
        underlying: Option<&str>,
        at: i64,
        settlement: bool,
    ) -> Result<&FeeRow> {
        let rows: Vec<&FeeRow> = self
            .rows
            .iter()
            .filter(|r| r.venue == venue && r.segment == segment && r.settlement == settlement)
            .collect();
        let own: Vec<&FeeRow> = rows
            .iter()
            .copied()
            .filter(|r| r.underlying.is_some() && r.underlying.as_deref() == underlying)
            .collect();
        let pool: Vec<&FeeRow> = if own.is_empty() {
            rows.iter().copied().filter(|r| r.underlying.is_none()).collect()
        } else {
            own
        };
        if pool.is_empty() {
            let suffix = underlying.map(|u| format!(" ({u})")).unwrap_or_default();
            return Err(RaptorError::invalid_config(format!(
                "{}: no rate for {venue} {}{suffix}",
                self.name,
                segment.as_str()
            )));
        }
        let live = pool.iter().copied().filter(|r| r.effective_from <= at);
        // First of equal keys, so ties are stable.
        let mut best: Option<&FeeRow> = None;
        for r in live {
            if best.is_none_or(|b| r.effective_from > b.effective_from) {
                best = Some(r);
            }
        }
        if let Some(b) = best {
            return Ok(b);
        }
        let mut earliest = pool[0];
        for r in &pool[1..] {
            if r.effective_from < earliest.effective_from {
                earliest = r;
            }
        }
        Ok(earliest)
    }

    /// Price a fill: pick the row and charge it. `value` is the turnover the
    /// rates apply to.
    #[allow(clippy::too_many_arguments)]
    pub fn charge(
        &self,
        venue: &str,
        segment: FeeSegment,
        underlying: Option<&str>,
        at: i64,
        is_buy: bool,
        value: f64,
        order_before: f64,
        day_fee_owed: bool,
    ) -> Result<FeeBreakdown> {
        let row = self.row_for(venue, segment, underlying, at, false)?;
        Ok(row.rates.charge_fill(is_buy, value.abs(), order_before, day_fee_owed))
    }
}

// --- the built-in exchange schedule --------------------------------------

/// Where the built-in rates come from.
pub const EXCHANGE_SOURCE: &str = "published exchange, clearing-corporation and regulator \
     schedules for NSE, BSE, NFO, BFO, MCX, CDS and BCD, checked 2026-08-20; STT and CTT as \
     revised 2026-04-01";

const GST: f64 = 0.18;
const SEBI: f64 = 0.000001;
/// Flat brokerage per executed order (also the cap where a rate applies).
const BROKERAGE: f64 = 20.0;
/// "0.03% or 20, whichever is lower".
const BROKERAGE_CAP_RATE: f64 = 0.0003;

/// Depository participant charge on a delivery sell, before GST: 13.00.
/// With 18% GST it is 15.34, owed once per ISIN per day with any sell, not
/// per order, which is why it is a [`DayFee`] and not a rate.
pub const DAY_FEE_DELIVERY_SELL: f64 = 13.0;

struct Spec {
    /// Flat brokerage; the cap when `rate` is set.
    flat: f64,
    rate: f64,
    tax_kind: TaxKind,
    tax_sell: f64,
    tax_buy: f64,
    exchange: f64,
    stamp_buy: f64,
    day_fee: Option<DayFee>,
}

impl Spec {
    const fn rates(&self) -> FeeRates {
        FeeRates {
            brokerage_flat: if self.rate > 0.0 { 0.0 } else { self.flat },
            brokerage_rate: self.rate,
            brokerage_cap: if self.rate > 0.0 { Some(self.flat) } else { None },
            brokerage_min: 0.0,
            brokerage_per: BrokeragePer::Fill,
            tax_kind: self.tax_kind,
            tax_buy_rate: self.tax_buy,
            tax_sell_rate: self.tax_sell,
            exchange_rate: self.exchange,
            regulatory_rate: SEBI,
            investor_fund_rate: 0.0,
            stamp_buy_rate: self.stamp_buy,
            gst_rate: GST,
            gst_base: GstBase::SERVICES,
            day_fee: self.day_fee,
            rounding: Rounding::None,
        }
    }
}

const fn sell_tax(kind: TaxKind, rate: f64) -> (TaxKind, f64, f64) {
    (kind, rate, 0.0)
}

const fn spec(
    flat: f64,
    rate: f64,
    tax: (TaxKind, f64, f64),
    exchange: f64,
    stamp_buy: f64,
    day_fee: Option<DayFee>,
) -> Spec {
    Spec {
        flat,
        rate,
        tax_kind: tax.0,
        tax_sell: tax.1,
        tax_buy: tax.2,
        exchange,
        stamp_buy,
        day_fee,
    }
}

const DELIVERY_DAY_FEE: DayFee =
    DayFee { amount: DAY_FEE_DELIVERY_SELL, side: DaySide::Sell, gst: true };

/// The rates of one built-in (venue, segment), or `None` where the venue
/// has no such segment. Venue names are the exchanges': nothing here is
/// named for a broker.
pub fn exchange_rates(venue: &str, segment: FeeSegment) -> Option<FeeRates> {
    use FeeSegment::*;
    let s = match (venue, segment) {
        // NSE/BSE equity. BSE shares the NSE-family schedule: its published
        // exchange-transaction rate differs by under 0.2 bps.
        ("NSE" | "BSE", Intraday) => spec(
            BROKERAGE,
            BROKERAGE_CAP_RATE,
            sell_tax(TaxKind::Stt, 0.00025),
            0.0000307,
            0.00003,
            None,
        ),
        ("NSE" | "BSE", Delivery) => {
            spec(0.0, 0.0, (TaxKind::Stt, 0.001, 0.001), 0.0000307, 0.00015, Some(DELIVERY_DAY_FEE))
        }
        // NFO/BFO derivatives. The STT on options *exercised* (0.15% of
        // intrinsic) is not modelled: positions close by trading out.
        ("NFO" | "BFO", Futures) => spec(
            BROKERAGE,
            BROKERAGE_CAP_RATE,
            sell_tax(TaxKind::Stt, 0.0005),
            0.0000183,
            0.00002,
            None,
        ),
        ("NFO" | "BFO", Options) => {
            spec(BROKERAGE, 0.0, sell_tax(TaxKind::Stt, 0.0015), 0.0003553, 0.00003, None)
        }
        ("MCX", Futures) => spec(
            BROKERAGE,
            BROKERAGE_CAP_RATE,
            sell_tax(TaxKind::Ctt, 0.0001),
            0.000021,
            0.00002,
            None,
        ),
        ("MCX", Options) => {
            spec(BROKERAGE, 0.0, sell_tax(TaxKind::Ctt, 0.0005), 0.000418, 0.00003, None)
        }
        // Currency derivatives: no transaction tax.
        ("CDS" | "BCD", Futures) => spec(
            BROKERAGE,
            BROKERAGE_CAP_RATE,
            (TaxKind::None, 0.0, 0.0),
            0.0000035,
            0.000001,
            None,
        ),
        ("CDS" | "BCD", Options) => {
            spec(BROKERAGE, 0.0, (TaxKind::None, 0.0, 0.0), 0.000311, 0.000001, None)
        }
        _ => return None,
    };
    Some(s.rates())
}

/// Every (venue, segment) the built-in schedule prices.
pub const EXCHANGE_PRESETS: &[(&str, FeeSegment)] = &[
    ("NSE", FeeSegment::Intraday),
    ("NSE", FeeSegment::Delivery),
    ("BSE", FeeSegment::Intraday),
    ("BSE", FeeSegment::Delivery),
    ("NFO", FeeSegment::Futures),
    ("NFO", FeeSegment::Options),
    ("BFO", FeeSegment::Futures),
    ("BFO", FeeSegment::Options),
    ("MCX", FeeSegment::Futures),
    ("MCX", FeeSegment::Options),
    ("CDS", FeeSegment::Futures),
    ("CDS", FeeSegment::Options),
    ("BCD", FeeSegment::Futures),
    ("BCD", FeeSegment::Options),
];

impl FeeSchedule {
    /// The built-in exchange schedule: the rows the engine charges when no
    /// schedule is supplied.
    pub fn exchange_presets() -> Self {
        let rows = EXCHANGE_PRESETS
            .iter()
            .map(|&(venue, segment)| FeeRow {
                venue: venue.to_string(),
                segment,
                underlying: None,
                settlement: false,
                effective_from: i64::MIN,
                rates: exchange_rates(venue, segment).expect("every listed preset has rates"),
            })
            .collect();
        FeeSchedule::new("exchange", EXCHANGE_SOURCE, rows)
    }
}
