//! A rate card: fees as a JSON document.
//!
//! ```json
//! {
//!   "name": "venue-card-2026-09",
//!   "effective_from": "2026-09-01T00:00:00+05:30",
//!   "source": "https://example.org/published-charges (read 2026-09-01)",
//!   "rates": [{
//!     "exchange": "NSE", "segment": "INTRADAY", "underlying": null,
//!     "brokerage": {"percent": "0.03", "cap": "20"},
//!     "transaction_tax": {"kind": "STT", "percent": "0.025", "sides": ["SELL"]},
//!     "exchange_turnover_percent": "0.00307", "sebi_percent": "0.0001",
//!     "stamp_duty_percent": "0.003", "gst_percent": "18"
//!   }]
//! }
//! ```
//!
//! Rates are percentages in decimal text (`"0.03"` is 0.03%), converted to
//! fractions without binary noise. A row prices as a contract note does
//! (`"rounding": "contract_note"`), the flat brokerage and its cap per order
//! (`"brokerage_per": "order"`), unless it says otherwise. Optional keys:
//! `brokerage.min`, `brokerage.per_order` together with `percent`,
//! `rounding` (`none`, `paise_each`, `contract_note`), `brokerage_per`
//! (`fill`, `order`), `investor_fund_percent`, `gst_base` (a list of
//! `brokerage`, `exchange`, `regulatory`, `investor_fund`,
//! `transaction_tax`, `stamp_duty`), `event` (`TRADE` or `SETTLEMENT`), and
//! `dp_charge` (`per_scrip_day`, optional `side`, `gst`) on a delivery row.
//! Unknown keys are ignored. The first failure is the error.

use serde_json::{json, Map, Value};

use super::rates::{BrokeragePer, DayFee, DaySide, FeeRates, GstBase, Rounding, TaxKind};
use super::schedule::{FeeRow, FeeSchedule, FeeSegment};
use crate::core::error::{RaptorError, Result};

const MINUTE_NS: i64 = 60_000_000_000;
const SECOND_NS: i64 = 1_000_000_000;

/// A validated rate card.
#[derive(Debug, Clone, PartialEq)]
pub struct RateCard {
    pub name: String,
    /// The document's `effective_from` text, as given.
    pub effective_from: String,
    pub effective_ns: i64,
    pub source: String,
    pub rows: Vec<FeeRow>,
}

fn refuse(message: String) -> RaptorError {
    RaptorError::invalid_parameter(message)
}

/// Days since 1970-01-01 of a proleptic Gregorian date.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// `YYYY-MM-DDTHH:MM:SS[.fraction](Z|+HH:MM)` as UTC nanoseconds.
pub fn parse_rfc3339_ns(text: &str) -> Result<i64> {
    let bad = || refuse(format!("{text:?} is not an RFC 3339 timestamp"));
    let b = text.as_bytes();
    if b.len() < 20 || !text.is_char_boundary(10) || !matches!(b[10], b'T' | b't' | b' ') {
        return Err(bad());
    }
    let num = |s: &str| -> Result<i64> {
        if s.is_empty() || !s.bytes().all(|c| c.is_ascii_digit()) {
            return Err(bad());
        }
        s.parse::<i64>().map_err(|_| bad())
    };
    let date = &text[..10];
    if date.as_bytes()[4] != b'-' || date.as_bytes()[7] != b'-' {
        return Err(bad());
    }
    let (y, m, d) = (num(&date[..4])?, num(&date[5..7])?, num(&date[8..10])?);
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let days_in = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    if !(1..=12).contains(&m) || d < 1 || d > days_in[(m - 1) as usize] {
        return Err(bad());
    }
    let rest = &text[11..];
    let (clock, offset) = if let Some(c) = rest.strip_suffix(['Z', 'z']) {
        (c, 0)
    } else {
        let at = rest.rfind(['+', '-']).ok_or_else(bad)?;
        let (c, off) = rest.split_at(at);
        let sign = if off.starts_with('-') { -1 } else { 1 };
        let hm = &off[1..];
        if hm.len() != 5 || hm.as_bytes()[2] != b':' {
            return Err(bad());
        }
        let (h, mi) = (num(&hm[..2])?, num(&hm[3..])?);
        if h > 23 || mi > 59 {
            return Err(bad());
        }
        (c, sign * (h * 60 + mi) * MINUTE_NS)
    };
    let (hms, frac) = clock.split_once('.').unwrap_or((clock, ""));
    if hms.len() != 8
        || hms.as_bytes()[2] != b':'
        || hms.as_bytes()[5] != b':'
        || frac.len() > 9
        || !frac.bytes().all(|c| c.is_ascii_digit())
    {
        return Err(bad());
    }
    let (hh, mm, ss) = (num(&hms[..2])?, num(&hms[3..5])?, num(&hms[6..])?);
    if hh > 23 || mm > 59 || ss > 59 {
        return Err(bad());
    }
    let nanos = if frac.is_empty() { 0 } else { num(&format!("{frac:0<9}"))? };
    Ok(days_from_civil(y, m, d) * 86_400 * SECOND_NS
        + (hh * 3600 + mm * 60 + ss) * SECOND_NS
        + nanos
        - offset)
}

/// A non-negative decimal from text or a JSON number, as a fraction when
/// `percent` (the text read two places down, so `"0.03"` is the double
/// nearest 0.0003, not 0.03 / 100).
fn decimal(where_: &str, v: Option<&Value>, percent: bool) -> Result<f64> {
    let text = match v {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Number(n)) => n.to_string(),
        other => {
            let shown = other.map_or("None".to_string(), Value::to_string);
            return Err(refuse(format!("{where_}: {shown} is not a decimal string")));
        }
    };
    let plain = {
        let mut parts = text.splitn(2, '.');
        let int = parts.next().unwrap_or("");
        let frac = parts.next();
        !int.is_empty()
            && int.bytes().all(|c| c.is_ascii_digit())
            && frac.is_none_or(|f| !f.is_empty() && f.bytes().all(|c| c.is_ascii_digit()))
    };
    if !plain {
        if text.starts_with('-') && text[1..].parse::<f64>().is_ok() {
            return Err(refuse(format!("{where_}: {text:?} is not a non-negative decimal")));
        }
        return Err(refuse(format!("{where_}: {text:?} is not a decimal string")));
    }
    let parsed = if percent { format!("{text}e-2") } else { text.clone() };
    parsed
        .parse::<f64>()
        .ok()
        .filter(|x| x.is_finite())
        .ok_or_else(|| refuse(format!("{where_}: {text:?} is not a decimal string")))
}

fn present(v: Option<&Value>) -> bool {
    !matches!(v, None | Some(Value::Null))
}

fn text(v: Option<&Value>) -> Option<&str> {
    match v {
        Some(Value::String(s)) => Some(s),
        _ => None,
    }
}

impl RateCard {
    /// Parse and validate a rate card document.
    pub fn from_json(doc: &str) -> Result<Self> {
        let doc: Value = serde_json::from_str(doc).map_err(|e| refuse(format!("not JSON: {e}")))?;
        let Value::Object(top) = &doc else {
            return Err(refuse("not a rate card document: expected an object".to_string()));
        };
        let name = match text(top.get("name")) {
            Some(s) if !s.is_empty() => s.to_string(),
            _ => return Err(refuse("the schedule has no name".to_string())),
        };
        let effective_from = match text(top.get("effective_from")) {
            Some(s) => s.to_string(),
            None => return Err(refuse(format!("{name}: effective_from is required"))),
        };
        let effective_ns = parse_rfc3339_ns(&effective_from)?;
        let rates = match top.get("rates") {
            Some(Value::Array(r)) if !r.is_empty() => r,
            _ => return Err(refuse(format!("{name}: rates must be a non-empty list"))),
        };
        let source = match top.get("source") {
            None | Some(Value::Null) => String::new(),
            Some(Value::String(s)) => s.clone(),
            Some(other) => other.to_string(),
        };
        let mut rows = Vec::with_capacity(rates.len());
        for (i, r) in rates.iter().enumerate() {
            rows.push(Self::row(&name, i, r, effective_ns)?);
        }
        Ok(RateCard { name, effective_from, effective_ns, source, rows })
    }

    fn row(name: &str, i: usize, r: &Value, effective_ns: i64) -> Result<FeeRow> {
        let w = format!("{name}: rates[{i}]");
        let Value::Object(r) = r else {
            return Err(refuse(format!("{w} is not an object")));
        };
        let segment =
            text(r.get("segment")).and_then(|s| FeeSegment::parse(s).ok()).ok_or_else(|| {
                let shown = r.get("segment").map_or("None".to_string(), Value::to_string);
                refuse(format!("{w}: unknown segment {shown}"))
            })?;
        let venue = match text(r.get("exchange")) {
            Some(s) if !s.is_empty() => s.to_string(),
            _ => return Err(refuse(format!("{w}: exchange is required"))),
        };
        let underlying = match r.get("underlying") {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) => Some(s.clone()),
            Some(other) => Some(other.to_string()),
        };
        let settlement = match r.get("event") {
            None | Some(Value::Null) => false,
            Some(Value::String(s)) if s == "TRADE" => false,
            Some(Value::String(s)) if s == "SETTLEMENT" => true,
            Some(other) => {
                return Err(refuse(format!("{w}.event: TRADE or SETTLEMENT, got {other}")))
            }
        };
        let empty = Map::new();
        let b = match r.get("brokerage") {
            Some(Value::Object(b)) => b,
            _ => &empty,
        };
        let (per_order, percent, cap) = (b.get("per_order"), b.get("percent"), b.get("cap"));
        if !present(per_order) && !present(percent) {
            return Err(refuse(format!("{w}.brokerage: per_order, or percent with cap")));
        }
        if present(cap) && !present(percent) {
            return Err(refuse(format!("{w}.brokerage: cap needs percent")));
        }
        let flat = if present(per_order) {
            decimal(&format!("{w}.brokerage.per_order"), per_order, false)?
        } else {
            0.0
        };
        let rate = if present(percent) {
            decimal(&format!("{w}.brokerage.percent"), percent, true)?
        } else {
            0.0
        };
        let cap = if present(cap) {
            Some(decimal(&format!("{w}.brokerage.cap"), cap, false)?)
        } else {
            None
        };
        let min = if present(b.get("min")) {
            decimal(&format!("{w}.brokerage.min"), b.get("min"), false)?
        } else {
            0.0
        };
        let tax = match r.get("transaction_tax") {
            Some(Value::Object(t)) => t,
            _ => &empty,
        };
        let tax_kind = match text(tax.get("kind")) {
            Some("STT") => TaxKind::Stt,
            Some("CTT") => TaxKind::Ctt,
            Some("NONE") => TaxKind::None,
            _ => return Err(refuse(format!("{w}.transaction_tax.kind: STT, CTT or NONE"))),
        };
        let zero_text = Value::String("0".to_string());
        let tax_rate = decimal(
            &format!("{w}.transaction_tax.percent"),
            Some(tax.get("percent").unwrap_or(&zero_text)),
            true,
        )?;
        let sides: Vec<&str> = match tax.get("sides") {
            Some(Value::Array(s)) => s.iter().filter_map(Value::as_str).collect(),
            _ => Vec::new(),
        };
        if tax_kind != TaxKind::None && sides.is_empty() {
            return Err(refuse(format!("{w}.transaction_tax: needs at least one side")));
        }
        let taxed = |side: &str, rate: f64| {
            if tax_kind != TaxKind::None && sides.contains(&side) {
                rate
            } else {
                0.0
            }
        };
        let field = |key: &str, percent: bool| decimal(&format!("{w}.{key}"), r.get(key), percent);
        let day_fee = match r.get("dp_charge") {
            None | Some(Value::Null) => None,
            Some(Value::Object(dp)) if segment == FeeSegment::Delivery && !settlement => {
                let amount = decimal(
                    &format!("{w}.dp_charge.per_scrip_day"),
                    dp.get("per_scrip_day"),
                    false,
                )?;
                let side = match text(dp.get("side")) {
                    None | Some("SELL") => DaySide::Sell,
                    Some("BUY") => DaySide::Buy,
                    Some(o) => {
                        return Err(refuse(format!("{w}.dp_charge.side: BUY or SELL, got {o:?}")))
                    }
                };
                let gst = match dp.get("gst") {
                    None | Some(Value::Null) => true,
                    Some(Value::Bool(g)) => *g,
                    Some(o) => {
                        return Err(refuse(format!("{w}.dp_charge.gst: true or false, got {o}")))
                    }
                };
                Some(DayFee { amount, side, gst })
            }
            Some(Value::Object(_)) => {
                return Err(refuse(format!("{w}.dp_charge: only a DELIVERY trade row has one")))
            }
            Some(other) => {
                return Err(refuse(format!(
                    "{w}.dp_charge: {{\"per_scrip_day\": ...}}, got {other}"
                )))
            }
        };
        let rounding = match text(r.get("rounding")) {
            None | Some("contract_note") => Rounding::ContractNote,
            Some("paise_each") => Rounding::PaiseEach,
            Some("none") => Rounding::None,
            Some(o) => {
                return Err(refuse(format!(
                    "{w}.rounding: none, paise_each or contract_note, got {o:?}"
                )))
            }
        };
        let brokerage_per = match text(r.get("brokerage_per")) {
            None | Some("order") => BrokeragePer::Order,
            Some("fill") => BrokeragePer::Fill,
            Some(o) => return Err(refuse(format!("{w}.brokerage_per: fill or order, got {o:?}"))),
        };
        let gst_base = match r.get("gst_base") {
            None | Some(Value::Null) => GstBase::SERVICES,
            Some(Value::Array(items)) => {
                let mut g = GstBase {
                    brokerage: false,
                    exchange: false,
                    regulatory: false,
                    investor_fund: false,
                    transaction_tax: false,
                    stamp_duty: false,
                };
                for item in items {
                    match item.as_str() {
                        Some("brokerage") => g.brokerage = true,
                        Some("exchange") => g.exchange = true,
                        Some("regulatory") => g.regulatory = true,
                        Some("investor_fund") => g.investor_fund = true,
                        Some("transaction_tax") => g.transaction_tax = true,
                        Some("stamp_duty") => g.stamp_duty = true,
                        _ => return Err(refuse(format!("{w}.gst_base: unknown component {item}"))),
                    }
                }
                g
            }
            Some(o) => return Err(refuse(format!("{w}.gst_base: a list of components, got {o}"))),
        };
        let investor_fund = if present(r.get("investor_fund_percent")) {
            field("investor_fund_percent", true)?
        } else {
            0.0
        };
        let rates = FeeRates {
            brokerage_flat: flat,
            brokerage_rate: rate,
            brokerage_cap: cap,
            brokerage_min: min,
            brokerage_per,
            tax_kind,
            tax_buy_rate: taxed("BUY", tax_rate),
            tax_sell_rate: taxed("SELL", tax_rate),
            exchange_rate: field("exchange_turnover_percent", true)?,
            regulatory_rate: field("sebi_percent", true)?,
            investor_fund_rate: investor_fund,
            stamp_buy_rate: field("stamp_duty_percent", true)?,
            gst_rate: field("gst_percent", true)?,
            gst_base,
            day_fee,
            rounding,
        };
        Ok(FeeRow { venue, segment, underlying, settlement, effective_from: effective_ns, rates })
    }

    /// The card as a schedule, named after it.
    pub fn fee_schedule(&self) -> FeeSchedule {
        FeeSchedule::new(self.name.clone(), self.source.clone(), self.rows.clone())
    }

    /// How many of `fill_ts` fell before the card took effect: those are
    /// priced on this card, not the one then in force.
    pub fn backdated_fills(&self, fill_ts: &[i64]) -> usize {
        fill_ts.iter().filter(|&&t| t < self.effective_ns).count()
    }

    /// The card as a document [`RateCard::from_json`] reads back to the same
    /// rows.
    pub fn to_json(&self) -> String {
        let rates: Vec<Value> = self.rows.iter().map(row_json).collect();
        json!({
            "name": self.name,
            "effective_from": self.effective_from,
            "source": self.source,
            "rates": rates,
        })
        .to_string()
    }
}

impl FeeSchedule {
    /// The schedule as a rate card document, for rows that share one
    /// effective date text.
    pub fn to_card_json(&self, effective_from: &str) -> Result<String> {
        let effective_ns = parse_rfc3339_ns(effective_from)?;
        let rows = self
            .rows
            .iter()
            .map(|r| FeeRow { effective_from: effective_ns, ..r.clone() })
            .collect();
        Ok(RateCard {
            name: self.name.clone(),
            effective_from: effective_from.to_string(),
            effective_ns,
            source: self.source.clone(),
            rows,
        }
        .to_json())
    }
}

/// `frac` as percent decimal text: the point moved two places by text, so no
/// binary noise shows (0.0003 reads `0.03`).
pub fn percent_text(frac: f64) -> String {
    let s = format!("{frac}");
    let (int, fr) = s.split_once('.').unwrap_or((&s, ""));
    let fr = format!("{fr:0<2}");
    let whole = format!("{int}{}", &fr[..2]);
    let whole = whole.trim_start_matches('0');
    let whole = if whole.is_empty() { "0" } else { whole };
    let rest = fr[2..].trim_end_matches('0');
    if rest.is_empty() {
        whole.to_string()
    } else {
        format!("{whole}.{rest}")
    }
}

fn plain_text(x: f64) -> String {
    format!("{x}")
}

fn row_json(row: &FeeRow) -> Value {
    let r = &row.rates;
    let mut brokerage = Map::new();
    if r.brokerage_flat > 0.0 || r.brokerage_rate == 0.0 {
        brokerage.insert("per_order".into(), plain_text(r.brokerage_flat).into());
    }
    if r.brokerage_rate > 0.0 {
        brokerage.insert("percent".into(), percent_text(r.brokerage_rate).into());
    }
    if let Some(cap) = r.brokerage_cap {
        brokerage.insert("cap".into(), plain_text(cap).into());
    }
    if r.brokerage_min > 0.0 {
        brokerage.insert("min".into(), plain_text(r.brokerage_min).into());
    }
    let mut sides = Vec::new();
    let mut tax_rate = 0.0;
    if r.tax_kind != TaxKind::None {
        if r.tax_buy_rate > 0.0 {
            sides.push("BUY");
            tax_rate = r.tax_buy_rate;
        }
        if r.tax_sell_rate > 0.0 {
            sides.push("SELL");
            tax_rate = r.tax_sell_rate;
        }
    }
    let tax_kind = if sides.is_empty() { "NONE" } else { r.tax_kind.as_str() };
    let mut out = json!({
        "exchange": row.venue,
        "segment": row.segment.as_str(),
        "underlying": row.underlying,
        "brokerage": Value::Object(brokerage),
        "transaction_tax": {
            "kind": tax_kind,
            "percent": percent_text(tax_rate),
            "sides": sides,
        },
        "exchange_turnover_percent": percent_text(r.exchange_rate),
        "sebi_percent": percent_text(r.regulatory_rate),
        "stamp_duty_percent": percent_text(r.stamp_buy_rate),
        "gst_percent": percent_text(r.gst_rate),
        "rounding": r.rounding.as_str(),
        "brokerage_per": r.brokerage_per.as_str(),
    });
    let o = out.as_object_mut().expect("object");
    if row.settlement {
        o.insert("event".into(), "SETTLEMENT".into());
    }
    if r.investor_fund_rate > 0.0 {
        o.insert("investor_fund_percent".into(), percent_text(r.investor_fund_rate).into());
    }
    let g = r.gst_base;
    if g != GstBase::SERVICES {
        let names = [
            (g.brokerage, "brokerage"),
            (g.exchange, "exchange"),
            (g.regulatory, "regulatory"),
            (g.investor_fund, "investor_fund"),
            (g.transaction_tax, "transaction_tax"),
            (g.stamp_duty, "stamp_duty"),
        ];
        let list: Vec<&str> = names.iter().filter(|(on, _)| *on).map(|(_, n)| *n).collect();
        o.insert("gst_base".into(), json!(list));
    }
    if let Some(day) = r.day_fee {
        o.insert(
            "dp_charge".into(),
            json!({
                "per_scrip_day": plain_text(day.amount),
                "side": if day.side == DaySide::Buy { "BUY" } else { "SELL" },
                "gst": day.gst,
            }),
        );
    }
    out
}

#[cfg(test)]
#[path = "card_tests.rs"]
mod tests;
