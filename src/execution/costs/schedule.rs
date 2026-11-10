//! Fee rows by venue, segment and underlying, and the schedule that picks
//! one for a fill.

use super::rates::{BrokeragePer, FeeBreakdown, FeeRates, GstBase, Rounding, TaxKind};
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

/// How far a row's rates can be trusted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RowStatus {
    /// Matched to an official circular or statute.
    Confirmed,
    /// From a secondary source, or a component the sources do not publish was
    /// left out: not to be taken as exchange-accurate. A result lists the
    /// unverified rows it used.
    Unverified,
    /// A user's own rate card: not judged here.
    #[default]
    Declared,
}

impl RowStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            RowStatus::Confirmed => "confirmed",
            RowStatus::Unverified => "unverified",
            RowStatus::Declared => "declared",
        }
    }
}

/// What the row's exchange charge and transaction tax are levied on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FeeBase {
    /// Traded value (price times quantity times multiplier).
    #[default]
    Turnover,
    /// Premium traded (an option's trade).
    Premium,
    /// Intrinsic value at exercise (an option settlement row).
    IntrinsicValue,
    /// Value of securities delivered.
    DeliveryValue,
}

impl FeeBase {
    pub fn as_str(self) -> &'static str {
        match self {
            FeeBase::Turnover => "turnover",
            FeeBase::Premium => "premium",
            FeeBase::IntrinsicValue => "intrinsic_value",
            FeeBase::DeliveryValue => "delivery_value",
        }
    }
}

/// What a row says about itself besides its rates.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RowMeta {
    pub status: RowStatus,
    pub base: FeeBase,
    /// UTC nanoseconds the rates stop applying at (exclusive); `None` while
    /// they are in force.
    pub effective_until: Option<i64>,
    /// The publication the rates come from, with its date.
    pub source: String,
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
    /// UTC nanoseconds. A fill before every row's date is refused
    /// (`fee_schedule_gap`), never priced on the earliest row.
    pub effective_from: i64,
    pub rates: FeeRates,
    pub meta: RowMeta,
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

/// `yyyy-mm-dd` of a UTC-ns instant on the Indian calendar (the day a row takes effect on).
fn ist_date_text(ns: i64) -> String {
    const IST_NS: i64 = (5 * 3600 + 30 * 60) * 1_000_000_000;
    const DAY_NS: i64 = 86_400_000_000_000;
    let ymd = crate::core::session_schedule::yyyymmdd_from_day(
        ns.saturating_add(IST_NS).div_euclid(DAY_NS),
    );
    format!("{:04}-{:02}-{:02}", ymd / 10_000, ymd / 100 % 100, ymd % 100)
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
    /// among rows in force at `at` the latest wins. No row is an error, never
    /// a free fill: a venue and segment the schedule has no row for, a date
    /// before the earliest row (named in the error), and a date after the
    /// last row ended.
    pub fn row_for(
        &self,
        venue: &str,
        segment: FeeSegment,
        underlying: Option<&str>,
        at: i64,
        settlement: bool,
    ) -> Result<&FeeRow> {
        match self.row_for_opt(venue, segment, underlying, at, settlement)? {
            Some(row) => Ok(row),
            None => {
                let suffix = underlying.map(|u| format!(" ({u})")).unwrap_or_default();
                Err(RaptorError::invalid_config(format!(
                    "{}: no rate for {venue} {}{suffix}",
                    self.name,
                    segment.as_str()
                )))
            }
        }
    }

    /// [`FeeSchedule::row_for`], telling "the schedule has no such row at all"
    /// (`Ok(None)`, which a settlement lookup reads as nothing to charge) from
    /// a row that exists but not for that date (an error: a date before the
    /// earliest row, or after the last one ended).
    pub fn row_for_opt(
        &self,
        venue: &str,
        segment: FeeSegment,
        underlying: Option<&str>,
        at: i64,
        settlement: bool,
    ) -> Result<Option<&FeeRow>> {
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
            return Ok(None);
        }
        let in_force =
            |r: &FeeRow| r.effective_from <= at && r.meta.effective_until.is_none_or(|u| at < u);
        // First of equal keys, so ties are stable.
        let mut best: Option<&FeeRow> = None;
        for r in pool.iter().copied().filter(|r| in_force(r)) {
            if best.is_none_or(|b| r.effective_from > b.effective_from) {
                best = Some(r);
            }
        }
        if let Some(b) = best {
            return Ok(Some(b));
        }
        let what = if settlement { "settlement" } else { segment.as_str() };
        if pool.iter().any(|r| r.effective_from <= at) {
            return Err(RaptorError::invalid_config(format!(
                "{}: no rate in force for {venue} {what} on {} (its rows have ended)",
                self.name,
                ist_date_text(at),
            )));
        }
        let earliest = pool.iter().map(|r| r.effective_from).min().unwrap_or(i64::MAX);
        Err(RaptorError::invalid_config(format!(
            "{}: no rate for {venue} {what} on {}: its earliest row takes effect on {}. A fill \
             before it would be priced on a rate that was not in force; load a schedule with \
             rows for that period",
            self.name,
            ist_date_text(at),
            ist_date_text(earliest),
        )))
    }

    /// This schedule with every row group (venue, segment, underlying, settlement) also priced
    /// before its earliest row, from `at` (UTC ns) on: a copy of that earliest row in force from
    /// `at` until the original takes effect, its source saying it is backdated and from when.
    /// A run over days no published rate covers is priced on the earliest rates known instead
    /// of being refused (`fee_schedule_gap`); a group whose earliest row is already in force by
    /// `at` gets nothing, so the rows that were in force price their own days as before.
    pub fn backdated(&self, at: i64) -> FeeSchedule {
        let mut rows = self.rows.clone();
        let mut done: Vec<(&str, FeeSegment, Option<&str>, bool)> = Vec::new();
        for r in &self.rows {
            let key = (r.venue.as_str(), r.segment, r.underlying.as_deref(), r.settlement);
            if done.contains(&key) {
                continue;
            }
            done.push(key);
            let Some(first) = self
                .rows
                .iter()
                .filter(|x| {
                    (x.venue.as_str(), x.segment, x.underlying.as_deref(), x.settlement) == key
                })
                .min_by_key(|x| x.effective_from)
            else {
                continue;
            };
            if first.effective_from <= at {
                continue;
            }
            let mut early = first.clone();
            early.meta.effective_until = Some(first.effective_from);
            early.meta.source = format!(
                "backdated: the rates in force from {} ({}), applied before any published rate",
                ist_date_text(first.effective_from),
                first.meta.source
            );
            early.effective_from = at;
            rows.push(early);
        }
        FeeSchedule::new(format!("{} (backdated)", self.name), self.source.clone(), rows)
    }

    /// The rows in force at `at` (UTC ns), one per venue, segment,
    /// underlying and event: the latest of each that has taken effect.
    /// Rows that have not taken effect yet are left out; this is the
    /// schedule as it stood that day, which a single-date rate card can carry.
    pub fn in_force_at(&self, at: i64) -> FeeSchedule {
        let mut keep: Vec<&FeeRow> = Vec::new();
        for row in self
            .rows
            .iter()
            .filter(|r| r.effective_from <= at && r.meta.effective_until.is_none_or(|u| at < u))
        {
            match keep.iter_mut().find(|k| {
                (&k.venue, k.segment, &k.underlying, k.settlement)
                    == (&row.venue, row.segment, &row.underlying, row.settlement)
            }) {
                Some(k) if row.effective_from > k.effective_from => *k = row,
                Some(_) => {}
                None => keep.push(row),
            }
        }
        FeeSchedule::new(
            self.name.clone(),
            self.source.clone(),
            keep.into_iter().cloned().collect(),
        )
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

/// Where the built-in rates come from. Each row names its own publication and
/// whether it was matched to an official one ([`RowStatus`]); a rate change is
/// a new dated row, never an edit.
pub const EXCHANGE_SOURCE: &str = "statutory and exchange charges only, in force from the dates \
     given, stored in rupees per crore and itemised: exchange transaction charge, investor \
     protection fund contribution (IPFT), SEBI turnover fee, transaction tax (STT, CTT), stamp \
     duty and GST. Rows matched to an official circular or statute are `confirmed`; the rest \
     (BSE cash by scrip group, every BSE derivatives row, NSE currency, MCX exchange charges, and \
     the NSE cash, futures and options rows before 2024-10-01, which use the slab-era exchange \
     rate and charge no IPFT) are `unverified` and a result lists the ones it used. The earliest \
     row is 2023-04-01 for NSE cash and NFO and 2024-10-01 for the other venues: a fill before a \
     venue and segment's first row is refused (`fee_schedule_gap`). No brokerage and no depository charge: \
     those come from a broker's rate card";

const GST: f64 = 0.18;
/// Rupees in a crore: the divisor from the stored rupees per crore to a fraction.
const CRORE: f64 = 1.0e7;

/// One component set in rupees per crore of the row's base.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Cr {
    tax_kind: TaxKind,
    tax_sell: f64,
    tax_buy: f64,
    exchange: f64,
    ipft: f64,
    sebi: f64,
    stamp_buy: f64,
}

impl Cr {
    const NONE: Cr = Cr {
        tax_kind: TaxKind::None,
        tax_sell: 0.0,
        tax_buy: 0.0,
        exchange: 0.0,
        ipft: 0.0,
        sebi: 0.0,
        stamp_buy: 0.0,
    };
}

/// A preset row before it is turned into a [`FeeRow`].
struct Stat {
    venue: &'static str,
    segment: FeeSegment,
    underlying: Option<&'static str>,
    settlement: bool,
    /// IST date (`yyyymmdd`) the rates take effect.
    from: i32,
    /// IST date the rates stop applying on (exclusive), if they have.
    until: Option<i32>,
    base: FeeBase,
    status: RowStatus,
    source: &'static str,
    cr: Cr,
}

use FeeSegment::{Delivery, Futures, Intraday, Options};

/// The earliest date any preset row takes effect: where the sourced statutory rates and the
/// slab-era exchange rates both begin. A fill before a (venue, segment)'s first row is refused.
const FIRST: i32 = 20230401;
/// The true-to-label exchange charges and the rows of every venue but NSE cash and NFO start here.
const START: i32 = 20241001;
const NSE_IPFT_SPLIT: i32 = 20260301;
const STT_2026: i32 = 20260401;

const SRC_NSE: &str = "NSE/FA/73061 (2026-02-27) and the existing-charge table it cites; SEBI \
     true-to-label structure from 2024-10-01";
const SRC_SLAB: &str = "slab-era representative rate: the NSE exchange transaction charge before \
     the true-to-label structure of 2024-10-01 (cash Rs 322, equity futures Rs 188, equity \
     options Rs 4,950 per crore), not matched to a circular; the transaction tax is statute \
     (Finance Act 2023, Finance (No. 2) Act 2024); IPFT for the period is not known and not \
     charged";
const SRC_STT: &str = "Finance Act 2026 (STT rates from 2026-04-01; Finance (No. 2) Act 2024 \
     before) - https://www.incometaxindia.gov.in";
const SRC_CTT: &str = "Commodities transaction tax, Finance Act 2013 (section 117), agricultural \
     commodities excluded - https://www.incometaxindia.gov.in";
const SRC_BSE_CUR: &str = "BSE Notice 20240927-36 (effective 2024-10-01)";
const UNVERIFIED_BSE_CASH: &str = "BSE equity notice 20221109-7 (group A/B); the current \
     group-wise schedule was not confirmed; IPFT not published in the sources and not charged";
const UNVERIFIED_BFO: &str = "BSE notice 20240927-37 as reported by a secondary source (Sensex \
     and Bankex options Rs 3,250/crore of premium, Sensex 50 and stock options Rs 500, futures \
     nil); not confirmed to the notice; IPFT not published and not charged";
const UNVERIFIED_CDS: &str = "NSE currency-derivatives true-to-label circular (Sep 2024) as \
     reported (futures 0.00035%, options 0.0311% of premium); not confirmed; IPFT not published \
     and not charged";
const UNVERIFIED_MCX: &str = "MCX/F&A/631/2024 (2024-09-24) as reported (futures Rs 2.10 per \
     lakh, options Rs 41.80 per lakh of premium); circular text not confirmed; IPFT not \
     published and not charged";

/// The key of the MCX agricultural fee row. Which contracts it covers is not listed here: a
/// contract is agricultural when [`crate::core::session_schedule::mcx_class`], the table that
/// picks its session, says so, so a name has one class for its hours and its charges.
pub const MCX_AGRI: &str = "class:AGRI";

/// The sell-side transaction tax of a row dated `date`, from `[until 2024-10-01, until
/// 2026-04-01, from 2026-04-01]`.
fn stt_cr(date: i32, by_era: [f64; 3]) -> f64 {
    if date >= STT_2026 {
        by_era[2]
    } else if date >= START {
        by_era[1]
    } else {
        by_era[0]
    }
}

/// The dates (ascending, from [`START`]) at which any of `edges` splits a period.
fn periods(edges: &[i32]) -> Vec<(i32, Option<i32>)> {
    periods_from(START, edges)
}

/// The dates (ascending, from `first`) at which any of `edges` splits a period.
fn periods_from(first: i32, edges: &[i32]) -> Vec<(i32, Option<i32>)> {
    let mut dates: Vec<i32> = std::iter::once(first).chain(edges.iter().copied()).collect();
    dates.sort_unstable();
    dates.dedup();
    dates.iter().enumerate().map(|(i, &d)| (d, dates.get(i + 1).copied())).collect()
}

fn statutory() -> Vec<Stat> {
    let mut rows = Vec::new();
    let mut push = |s: Stat| rows.push(s);
    // NSE cash. Exchange 297 + IPFT 10 per crore until 2026-02-28, 306.99 + 0.01 after; before
    // 2024-10-01 the slab-era 322 and no IPFT.
    for (segment, tax_sell, tax_buy, stamp) in
        [(Delivery, 10_000.0, 10_000.0, 1_500.0), (Intraday, 2_500.0, 0.0, 300.0)]
    {
        for (from, until) in periods_from(FIRST, &[START, NSE_IPFT_SPLIT]) {
            let (exchange, ipft, status, source) = if from < START {
                (322.0, 0.0, RowStatus::Unverified, SRC_SLAB)
            } else if from >= NSE_IPFT_SPLIT {
                (306.99, 0.01, RowStatus::Confirmed, SRC_NSE)
            } else {
                (297.0, 10.0, RowStatus::Confirmed, SRC_NSE)
            };
            push(Stat {
                venue: "NSE",
                segment,
                underlying: None,
                settlement: false,
                from,
                until,
                base: FeeBase::Turnover,
                status,
                source,
                cr: Cr {
                    tax_kind: TaxKind::Stt,
                    tax_sell,
                    tax_buy,
                    exchange,
                    ipft,
                    sebi: 10.0,
                    stamp_buy: stamp,
                },
            });
        }
    }
    // BSE cash, by scrip group (the group is part of the row's key).
    for group in ["group:A", "group:B"] {
        for (segment, tax_sell, tax_buy, stamp) in
            [(Delivery, 10_000.0, 10_000.0, 1_500.0), (Intraday, 2_500.0, 0.0, 300.0)]
        {
            push(Stat {
                venue: "BSE",
                segment,
                underlying: Some(group),
                settlement: false,
                from: START,
                until: None,
                base: FeeBase::Turnover,
                status: RowStatus::Unverified,
                source: UNVERIFIED_BSE_CASH,
                cr: Cr {
                    tax_kind: TaxKind::Stt,
                    tax_sell,
                    tax_buy,
                    exchange: 375.0,
                    ipft: 0.0,
                    sebi: 10.0,
                    stamp_buy: stamp,
                },
            });
        }
    }
    // NSE equity derivatives. Sell-side STT: futures 0.0125% from 2023-04-01, 0.02% from
    // 2024-10-01, 0.05% from 2026-04-01; options premium 0.0625%, 0.1%, 0.15%. Before 2024-10-01
    // the exchange charge is the slab-era rate (futures 188, options 4,950) and IPFT is not known.
    for (from, until) in periods_from(FIRST, &[START, NSE_IPFT_SPLIT, STT_2026]) {
        let split = from >= NSE_IPFT_SPLIT;
        let slab = from < START;
        let (status, source) =
            if slab { (RowStatus::Unverified, SRC_SLAB) } else { (RowStatus::Confirmed, SRC_NSE) };
        push(Stat {
            venue: "NFO",
            segment: Futures,
            underlying: None,
            settlement: false,
            from,
            until,
            base: FeeBase::Turnover,
            status,
            source,
            cr: Cr {
                tax_kind: TaxKind::Stt,
                tax_sell: stt_cr(from, [1_250.0, 2_000.0, 5_000.0]),
                tax_buy: 0.0,
                exchange: if slab {
                    188.0
                } else if split {
                    182.99
                } else {
                    173.0
                },
                ipft: if slab {
                    0.0
                } else if split {
                    0.01
                } else {
                    10.0
                },
                sebi: 10.0,
                stamp_buy: 200.0,
            },
        });
        push(Stat {
            venue: "NFO",
            segment: Options,
            underlying: None,
            settlement: false,
            from,
            until,
            base: FeeBase::Premium,
            status,
            source,
            cr: Cr {
                tax_kind: TaxKind::Stt,
                tax_sell: stt_cr(from, [6_250.0, 10_000.0, 15_000.0]),
                tax_buy: 0.0,
                exchange: if slab {
                    4_950.0
                } else if split {
                    3_552.0
                } else {
                    3_503.0
                },
                ipft: if slab {
                    0.0
                } else if split {
                    0.01
                } else {
                    50.0
                },
                sebi: 10.0,
                stamp_buy: 300.0,
            },
        });
    }
    // The tax on an option exercised at expiry, on its intrinsic value, charged to the buyer:
    // 0.125% until 2026-03-31, 0.15% after. NFO from 2023-04-01, BFO with its other rows.
    for venue in ["NFO", "BFO"] {
        let first = if venue == "NFO" { FIRST } else { START };
        for (from, until) in periods_from(first, &[STT_2026]) {
            push(Stat {
                venue,
                segment: Options,
                underlying: None,
                settlement: true,
                from,
                until,
                base: FeeBase::IntrinsicValue,
                status: RowStatus::Confirmed,
                source: SRC_STT,
                cr: Cr {
                    tax_kind: TaxKind::Stt,
                    tax_sell: if from >= STT_2026 { 15_000.0 } else { 12_500.0 },
                    ..Cr::NONE
                },
            });
        }
    }
    // BSE equity derivatives: every exchange figure here is unverified.
    for (from, until) in periods(&[STT_2026]) {
        push(Stat {
            venue: "BFO",
            segment: Futures,
            underlying: None,
            settlement: false,
            from,
            until,
            base: FeeBase::Turnover,
            status: RowStatus::Unverified,
            source: UNVERIFIED_BFO,
            cr: Cr {
                tax_kind: TaxKind::Stt,
                tax_sell: stt_cr(from, [2_000.0, 2_000.0, 5_000.0]),
                sebi: 10.0,
                stamp_buy: 200.0,
                ..Cr::NONE
            },
        });
        for (underlying, exchange) in [
            (None, 500.0),
            (Some("SENSEX"), 3_250.0),
            (Some("BANKEX"), 3_250.0),
            (Some("SENSEX50"), 500.0),
        ] {
            push(Stat {
                venue: "BFO",
                segment: Options,
                underlying,
                settlement: false,
                from,
                until,
                base: FeeBase::Premium,
                status: RowStatus::Unverified,
                source: UNVERIFIED_BFO,
                cr: Cr {
                    tax_kind: TaxKind::Stt,
                    tax_sell: stt_cr(from, [10_000.0, 10_000.0, 15_000.0]),
                    exchange,
                    sebi: 10.0,
                    stamp_buy: 300.0,
                    ..Cr::NONE
                },
            });
        }
    }
    // Currency derivatives: stamp duty 0.0001%, no transaction tax.
    for (venue, segment, exchange, status, source) in [
        ("CDS", Futures, 35.0, RowStatus::Unverified, UNVERIFIED_CDS),
        ("CDS", Options, 3_110.0, RowStatus::Unverified, UNVERIFIED_CDS),
        ("BCD", Futures, 45.0, RowStatus::Confirmed, SRC_BSE_CUR),
        ("BCD", Options, 100.0, RowStatus::Confirmed, SRC_BSE_CUR),
    ] {
        push(Stat {
            venue,
            segment,
            underlying: None,
            settlement: false,
            from: START,
            until: None,
            base: if segment == Options { FeeBase::Premium } else { FeeBase::Turnover },
            status,
            source,
            cr: Cr { exchange, sebi: 10.0, stamp_buy: 10.0, ..Cr::NONE },
        });
    }
    // MCX. The exchange charges are unverified; the tax is statute.
    push(Stat {
        venue: "MCX",
        segment: Futures,
        underlying: None,
        settlement: false,
        from: START,
        until: None,
        base: FeeBase::Turnover,
        status: RowStatus::Unverified,
        source: UNVERIFIED_MCX,
        cr: Cr {
            tax_kind: TaxKind::Ctt,
            tax_sell: 1_000.0,
            exchange: 210.0,
            sebi: 10.0,
            stamp_buy: 200.0,
            ..Cr::NONE
        },
    });
    push(Stat {
        venue: "MCX",
        segment: Futures,
        underlying: Some(MCX_AGRI),
        settlement: false,
        from: START,
        until: None,
        base: FeeBase::Turnover,
        status: RowStatus::Unverified,
        source: UNVERIFIED_MCX,
        cr: Cr { exchange: 210.0, sebi: 1.0, stamp_buy: 200.0, ..Cr::NONE },
    });
    push(Stat {
        venue: "MCX",
        segment: Options,
        underlying: None,
        settlement: false,
        from: START,
        until: None,
        base: FeeBase::Premium,
        status: RowStatus::Unverified,
        source: UNVERIFIED_MCX,
        cr: Cr {
            tax_kind: TaxKind::Ctt,
            tax_sell: 5_000.0,
            exchange: 4_180.0,
            sebi: 10.0,
            stamp_buy: 300.0,
            ..Cr::NONE
        },
    });
    push(Stat {
        venue: "MCX",
        segment: Options,
        underlying: None,
        settlement: true,
        from: START,
        until: None,
        base: FeeBase::IntrinsicValue,
        status: RowStatus::Confirmed,
        source: SRC_CTT,
        cr: Cr { tax_kind: TaxKind::Ctt, tax_sell: 10.0, ..Cr::NONE },
    });
    let _ = SRC_CTT;
    rows
}

fn rates_of(row: &Stat) -> FeeRates {
    let c = row.cr;
    let gst_base = GstBase { investor_fund: true, ..GstBase::SERVICES };
    FeeRates {
        brokerage_flat: 0.0,
        brokerage_rate: 0.0,
        brokerage_cap: None,
        brokerage_min: 0.0,
        brokerage_per: BrokeragePer::Fill,
        tax_kind: c.tax_kind,
        tax_buy_rate: c.tax_buy / CRORE,
        tax_sell_rate: c.tax_sell / CRORE,
        exchange_rate: c.exchange / CRORE,
        regulatory_rate: c.sebi / CRORE,
        investor_fund_rate: c.ipft / CRORE,
        stamp_buy_rate: c.stamp_buy / CRORE,
        // A settlement row owes the tax alone: no GST base without a charge.
        gst_rate: if row.settlement { 0.0 } else { GST },
        gst_base,
        day_fee: None,
        rounding: Rounding::None,
    }
}

/// UTC nanoseconds of 00:00 IST on a `yyyymmdd` date.
fn ist_midnight_ns(date: i32) -> i64 {
    const IST_NS: i64 = (5 * 3600 + 30 * 60) * 1_000_000_000;
    let day = crate::core::session_schedule::day_from_yyyymmdd(date)
        .expect("a preset date is a real date");
    day * 86_400_000_000_000 - IST_NS
}

fn fee_row_of(r: &Stat) -> FeeRow {
    FeeRow {
        venue: r.venue.to_string(),
        segment: r.segment,
        underlying: r.underlying.map(str::to_string),
        settlement: r.settlement,
        effective_from: ist_midnight_ns(r.from),
        rates: rates_of(r),
        meta: RowMeta {
            status: r.status,
            base: r.base,
            effective_until: r.until.map(ist_midnight_ns),
            source: r.source.to_string(),
        },
    }
}

/// The rates of one built-in (venue, segment) in force now (the latest
/// dated general row), or `None` where the venue has no such segment. Venue
/// names are the exchanges': nothing here is named for a broker. Use
/// [`FeeSchedule::exchange_presets`] and its dated rows to price a fill by
/// its own date.
pub fn exchange_rates(venue: &str, segment: FeeSegment) -> Option<FeeRates> {
    statutory()
        .iter()
        .filter(|r| {
            r.venue == venue && r.segment == segment && r.underlying.is_none() && !r.settlement
        })
        .max_by_key(|r| r.from)
        .map(rates_of)
}

/// What an option settled in cash costs the holder at expiry, as a settlement
/// row: the STT on options *exercised* (0.125% of the intrinsic value until
/// 2026-03-31, 0.15% after: this is the rate in force now), charged to
/// the buyer (the line closing a long) and to no one else, nothing else. An
/// option that expires worthless settles at an intrinsic value of zero, so it
/// owes nothing. `None` for a venue with no such tax. A physically settled
/// contract does not use this row: its delivery is a separate event charged at
/// the delivery rates of the underlying.
pub fn exercise_rates(venue: &str) -> Option<FeeRates> {
    statutory()
        .iter()
        .filter(|r| r.venue == venue && r.segment == Options && r.settlement)
        .max_by_key(|r| r.from)
        .map(rates_of)
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
    /// The built-in exchange schedule: the dated statutory rows the engine
    /// charges when no schedule is supplied. No brokerage: the schedule is
    /// what the exchanges, the tax authority and the regulator levy, and a
    /// broker's own charges come from its rate card.
    pub fn exchange_presets() -> Self {
        let rows = statutory().iter().map(fee_row_of).collect();
        FeeSchedule::new("exchange", EXCHANGE_SOURCE, rows)
    }
}
