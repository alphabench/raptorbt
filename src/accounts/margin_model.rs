//! Margin as pluggable models over open lines.
//!
//! `required(line)` is the initial margin a line demands. These are named
//! models with typed parameters, plus [`MarginModel::Table`] for per-instrument
//! figures that come from outside (an exchange's own span numbers, say).
//!
//! One path serves the engine too. [`MarginModel::FromSpec`] is the rule the
//! kernel applies on a margin account: the instrument's own `margin_init`
//! (else one over the account's leverage) on notional, and for a SOLD option
//! with `span_pct`/`exposure_pct` the per-contract deposit of
//! [`InstrumentSpec::short_option_margin_per_contract`], with the sold legs of
//! one underlying and expiry margined together by
//! [`crate::portfolio::option_groups`]. The kernel takes its rate from
//! [`initial_rate`] and its deposit from the spec method, and `FromSpec`
//! calls the same two, so a number cannot be defined twice.
//!
//! Margin is `f64` and unrounded: it gates entries and marks maintenance, it
//! is never a charge on a note (see `docs/quantitative-correctness/numerics.md`).

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};

use crate::core::error::{RaptorError, Result};
use crate::core::types::{Direction, Timestamp};
use crate::instruments::{InstrumentKind, InstrumentSpec, OptionRight};
use crate::portfolio::option_groups::{apportion, group_requirement, OptionLeg};

const DAY_NS: i64 = 86_400_000_000_000;

/// The calendar day of a timestamp in the zone `tz_offset_ns` ahead of UTC,
/// as days since the epoch.
fn local_day(ts: Timestamp, tz_offset_ns: i64) -> i64 {
    (ts + tz_offset_ns).div_euclid(DAY_NS)
}

/// The zone an instrument's days are read in: its session schedule's, else
/// UTC. No market's zone is assumed.
fn zone_of(spec: &InstrumentSpec) -> i64 {
    spec.session.as_ref().map_or(0, |s| s.tz_offset_ns())
}

/// The rate of initial margin on notional the engine uses when an account
/// funds positions on margin: the instrument's own `margin_init` when set,
/// else one over the account's leverage (never below 1.0).
#[inline]
pub fn initial_rate(spec: Option<&InstrumentSpec>, leverage: f64) -> f64 {
    spec.map(|s| s.margin_init).filter(|&m| m > 0.0).unwrap_or(1.0 / leverage.max(1.0))
}

/// The class of instrument a rule is written for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MarginClass {
    /// Equity and other cash instruments.
    Cash,
    CurrencyPair,
    /// Futures and perpetuals.
    Contract,
    Option,
}

impl MarginClass {
    pub fn as_str(self) -> &'static str {
        match self {
            MarginClass::Cash => "CASH",
            MarginClass::CurrencyPair => "CURRENCY_PAIR",
            MarginClass::Contract => "CONTRACT",
            MarginClass::Option => "OPTION",
        }
    }

    /// A class by name. `EQUITY` and `ETF` read as `CASH`, `FUTURE` and
    /// `PERPETUAL` as `CONTRACT`, `CURRENCY` as `CURRENCY_PAIR`.
    pub fn parse(text: &str) -> Result<Self> {
        Ok(match text.to_ascii_uppercase().as_str() {
            "CASH" | "EQUITY" | "ETF" => MarginClass::Cash,
            "CURRENCY_PAIR" | "CURRENCY" => MarginClass::CurrencyPair,
            "CONTRACT" | "FUTURE" | "FUTURES" | "PERPETUAL" => MarginClass::Contract,
            "OPTION" => MarginClass::Option,
            other => {
                return Err(invalid(format!(
                    "unknown instrument class {other:?}: CASH, CURRENCY_PAIR, CONTRACT or OPTION"
                )))
            }
        })
    }

    /// The class of `spec`. An index is not tradable and has no margin.
    pub fn of(spec: &InstrumentSpec) -> Result<Self> {
        match spec.kind {
            InstrumentKind::Cash => Ok(MarginClass::Cash),
            InstrumentKind::CurrencyPair => Ok(MarginClass::CurrencyPair),
            InstrumentKind::Contract { .. } => Ok(MarginClass::Contract),
            InstrumentKind::Option { .. } => Ok(MarginClass::Option),
            InstrumentKind::Index => {
                Err(invalid(format!("{}: an index is not tradable and has no margin", spec.symbol)))
            }
        }
    }
}

/// One open line as a margin model sees it.
#[derive(Debug, Clone, Copy)]
pub struct LineView<'a> {
    pub instrument: &'a InstrumentSpec,
    /// Signed: negative is short.
    pub quantity: f64,
    /// The mark.
    pub price: f64,
}

impl<'a> LineView<'a> {
    pub fn new(instrument: &'a InstrumentSpec, quantity: f64, price: f64) -> Self {
        LineView { instrument, quantity, price }
    }

    fn notional(&self) -> f64 {
        (self.price * self.quantity * self.instrument.multiplier).abs()
    }
}

/// A margin model.
#[derive(Debug, Clone, PartialEq)]
pub enum MarginModel {
    /// Nothing is required, nothing is refused.
    NoMargin,
    /// Delivery: the full notional of a long; a short is refused (infinite).
    CashOnly,
    /// Notional over a leverage factor per instrument class.
    Leverage(BTreeMap<MarginClass, f64>),
    /// A percentage of the notional.
    PercentOfNotional(f64),
    /// A buyer pays the premium; a seller posts `seller_pct` of the strike
    /// notional (the strike as the proxy for the underlying) plus the
    /// premium.
    OptionSimple { seller_pct: f64 },
    /// Per-unit margin per instrument symbol; a missing symbol falls back to
    /// `default` or is refused.
    Table { per_unit: BTreeMap<String, f64>, default: Option<Box<MarginModel>> },
    /// Each instrument class on its own model.
    ByType(BTreeMap<MarginClass, MarginModel>),
    /// The intraday (MIS) quantity of a line on `intraday`, the rest (CNC,
    /// NRML) on `carry`.
    ByProduct { intraday: Box<MarginModel>, carry: Box<MarginModel> },
    /// The worst expiry loss of a set of option legs
    /// ([`MarginModel::required_legs`]); one line is a set of one.
    DefinedRisk,
    /// `OptionSimple { seller_pct }` for every option line, less a credit for
    /// the lines of one underlying that cover each other: a defined-risk
    /// structure within one expiry gives up to its max loss
    /// ([`spread_credit_shares`], rule (b)), and a short option covered by a
    /// long option of the same right in a later expiry (a calendar or
    /// diagonal) gives a calendar credit that leaves at least `spread_pct`
    /// of the short's strike notional (rule (c)), removed on the short's
    /// expiry day (rule (d)). A naked or unhedged leg is `OptionSimple`
    /// exactly. Grouped across expiries, so it needs the time
    /// ([`MarginModel::required_legs_at`]).
    SpreadCredit { seller_pct: f64, spread_pct: f64 },
    /// The engine's own rule, read from each instrument: its `margin_init`
    /// (else one over `leverage`) on notional, and the span-and-exposure
    /// deposit for a sold option. Sold options of one underlying and expiry
    /// are margined together as a position group
    /// ([`crate::portfolio::option_groups`]).
    FromSpec { leverage: f64 },
}

fn invalid(message: String) -> RaptorError {
    RaptorError::invalid_parameter(message)
}

impl MarginModel {
    pub fn name(&self) -> &'static str {
        match self {
            MarginModel::NoMargin => "Margin.none",
            MarginModel::CashOnly => "Margin.cash_only",
            MarginModel::Leverage(_) => "Margin.leverage",
            MarginModel::PercentOfNotional(_) => "Margin.percent_of_notional",
            MarginModel::OptionSimple { .. } => "Margin.option_simple",
            MarginModel::Table { .. } => "Margin.table",
            MarginModel::ByType(_) => "Margin.by_type",
            MarginModel::ByProduct { .. } => "Margin.by_product",
            MarginModel::DefinedRisk => "Margin.defined_risk",
            MarginModel::SpreadCredit { .. } => "Margin.spread_credit",
            MarginModel::FromSpec { .. } => "Margin.from_spec",
        }
    }

    /// The initial margin `line` demands. A class the model has no rule for
    /// is an error, never a zero.
    pub fn required(&self, line: &LineView<'_>) -> Result<f64> {
        Ok(match self {
            MarginModel::NoMargin => 0.0,
            MarginModel::CashOnly => {
                if line.quantity < 0.0 {
                    f64::INFINITY
                } else {
                    line.price * line.quantity * line.instrument.multiplier
                }
            }
            MarginModel::Leverage(factors) => {
                let class = MarginClass::of(line.instrument)?;
                match factors.get(&class) {
                    Some(&x) if x > 0.0 => line.notional() / x,
                    _ => return Err(invalid(format!("no leverage for {}", class.as_str()))),
                }
            }
            MarginModel::PercentOfNotional(pct) => line.notional() * pct / 100.0,
            MarginModel::OptionSimple { seller_pct }
            | MarginModel::SpreadCredit { seller_pct, .. } => option_simple(line, *seller_pct),
            MarginModel::Table { per_unit, default } => {
                match (per_unit.get(&line.instrument.symbol), default) {
                    (Some(m), _) => m * line.quantity.abs(),
                    (None, Some(d)) => d.required(line)?,
                    (None, None) => {
                        return Err(invalid(format!("no margin for {}", line.instrument.symbol)))
                    }
                }
            }
            MarginModel::ByType(models) => {
                let class = MarginClass::of(line.instrument)?;
                match models.get(&class) {
                    Some(m) => m.required(line)?,
                    None => return Err(invalid(format!("no margin model for {}", class.as_str()))),
                }
            }
            // The whole line as carry: a caller that cannot split by product
            // gets the stricter delivery figure.
            MarginModel::ByProduct { carry, .. } => carry.required(line)?,
            MarginModel::DefinedRisk => defined_risk(std::slice::from_ref(line))?,
            MarginModel::FromSpec { leverage } => {
                let spec = line.instrument;
                let deposit = if line.quantity < 0.0 {
                    spec.short_option_margin_per_contract()
                } else {
                    None
                };
                match deposit {
                    Some(per_contract) => per_contract * line.quantity.abs(),
                    None => line.notional() * initial_rate(Some(spec), *leverage),
                }
            }
        })
    }

    /// `ByProduct`: the intraday part of the line on `intraday`, the rest on
    /// `carry`. Any other model prices the whole line.
    pub fn required_split(&self, line: &LineView<'_>, intraday_quantity: f64) -> Result<f64> {
        let MarginModel::ByProduct { intraday, carry } = self else {
            return self.required(line);
        };
        let carry_q = line.quantity - intraday_quantity;
        let mut total = 0.0;
        if intraday_quantity != 0.0 {
            total += intraday.required(&LineView { quantity: intraday_quantity, ..*line })?;
        }
        if carry_q != 0.0 {
            total += carry.required(&LineView { quantity: carry_q, ..*line })?;
        }
        Ok(total)
    }

    /// Whether a line of class `class` is margined by its worst expiry loss
    /// (`DefinedRisk`, or `ByType` giving the class `DefinedRisk`): then the
    /// lines of one underlying and expiry are margined together.
    pub fn is_defined_risk_for(&self, class: MarginClass) -> bool {
        match self {
            MarginModel::DefinedRisk => true,
            MarginModel::ByType(models) => {
                matches!(models.get(&class), Some(MarginModel::DefinedRisk))
            }
            _ => false,
        }
    }

    /// `DefinedRisk`: the worst expiry loss of the legs together. `FromSpec`:
    /// the sold legs as one position group, the bought legs at their own
    /// figure. Any other model sums the legs. `SpreadCredit` needs the time:
    /// use [`MarginModel::required_legs_at`].
    pub fn required_legs(&self, legs: &[LineView<'_>]) -> Result<f64> {
        match self {
            MarginModel::DefinedRisk => defined_risk(legs),
            MarginModel::SpreadCredit { .. } => {
                Err(invalid("spread_credit needs the time: use required_legs_at".to_string()))
            }
            MarginModel::FromSpec { .. } => Ok(self.group_shares(legs, 0)?.iter().sum()),
            _ => legs.iter().map(|l| self.required(l)).sum(),
        }
    }

    /// [`MarginModel::required_legs`] at `now` (the time the calendar credit
    /// of `SpreadCredit` is judged at).
    pub fn required_legs_at(&self, legs: &[LineView<'_>], now: Timestamp) -> Result<f64> {
        match self {
            MarginModel::SpreadCredit { seller_pct, spread_pct } => {
                Ok(spread_credit_shares(legs, *seller_pct, *spread_pct, now)?.iter().sum())
            }
            _ => self.required_legs(legs),
        }
    }

    /// The model that margins the option lines of an underlying together
    /// (this one, or the `OPTION` entry of a `ByType`), when there is one.
    fn option_group_model(&self) -> Option<&MarginModel> {
        match self {
            MarginModel::DefinedRisk
            | MarginModel::SpreadCredit { .. }
            | MarginModel::FromSpec { .. } => Some(self),
            MarginModel::ByType(models) => match models.get(&MarginClass::Option) {
                Some(
                    m @ (MarginModel::DefinedRisk
                    | MarginModel::SpreadCredit { .. }
                    | MarginModel::FromSpec { .. }),
                ) => Some(m),
                _ => None,
            },
            _ => None,
        }
    }

    /// Whether the option lines are margined together at all (`DefinedRisk`,
    /// `SpreadCredit` or `FromSpec`, directly or as the `OPTION` entry of a
    /// `ByType`).
    pub fn groups_options(&self) -> bool {
        self.option_group_model().is_some()
    }

    /// Whether the group reaches across expiries (`SpreadCredit`): then it
    /// is the lines of one underlying, not of one underlying and expiry.
    pub fn groups_across_expiries(&self) -> bool {
        matches!(self.option_group_model(), Some(MarginModel::SpreadCredit { .. }))
    }

    /// What a group of option lines locks together at `now`, as a sum of
    /// [`MarginModel::group_shares`].
    pub fn group_required(&self, legs: &[LineView<'_>], now: Timestamp) -> Result<f64> {
        Ok(self.group_shares(legs, now)?.iter().sum())
    }

    /// What each line of a group locks at `now`, in the order of `legs`.
    /// `DefinedRisk`: the group's worst loss shared over the lines in
    /// proportion to each line's own. `SpreadCredit`: each line's own
    /// figure, the credits taken off the short lines they cover.
    /// `FromSpec`: the sold legs share the position group's requirement in
    /// proportion to what each would carry alone; a bought leg keeps its own.
    pub fn group_shares(&self, legs: &[LineView<'_>], now: Timestamp) -> Result<Vec<f64>> {
        match self.option_group_model() {
            Some(MarginModel::SpreadCredit { seller_pct, spread_pct }) => {
                spread_credit_shares(legs, *seller_pct, *spread_pct, now)
            }
            Some(m @ MarginModel::FromSpec { .. }) => from_spec_shares(m, legs),
            Some(_) => {
                let total = defined_risk(legs)?;
                if total.is_infinite() {
                    // No bound on the loss: every line of the group is refused.
                    return Ok(vec![f64::INFINITY; legs.len()]);
                }
                // Each line's weight in the shared figure is its own worst
                // loss; a line with no bound alone (a short call covered by
                // another leg) weighs its strike notional instead.
                let naked: Vec<f64> = legs
                    .iter()
                    .map(|v| {
                        let alone = defined_risk(std::slice::from_ref(v)).unwrap_or(0.0).max(0.0);
                        match v.instrument.kind {
                            InstrumentKind::Option { strike, .. } if alone.is_infinite() => {
                                strike * v.quantity.abs() * v.instrument.multiplier
                            }
                            _ => alone,
                        }
                    })
                    .collect();
                let sum: f64 = naked.iter().sum();
                Ok((0..legs.len())
                    .map(
                        |k| {
                            if sum > 0.0 {
                                total * naked[k] / sum
                            } else {
                                total / legs.len() as f64
                            }
                        },
                    )
                    .collect())
            }
            None => Err(invalid("the model does not group option lines".to_string())),
        }
    }

    /// Parameters, with nested models given by name only.
    pub fn params(&self) -> Value {
        let mut out = Map::new();
        match self {
            MarginModel::NoMargin | MarginModel::CashOnly | MarginModel::DefinedRisk => {}
            MarginModel::SpreadCredit { seller_pct, spread_pct } => {
                out.insert("seller_pct".into(), json!(seller_pct));
                out.insert("spread_pct".into(), json!(spread_pct));
            }
            MarginModel::FromSpec { leverage } => {
                out.insert("leverage".into(), json!(leverage));
            }
            MarginModel::Leverage(f) => {
                for (k, v) in f {
                    out.insert(k.as_str().to_string(), json!(v));
                }
            }
            MarginModel::PercentOfNotional(p) => {
                out.insert("pct".into(), json!(p));
            }
            MarginModel::OptionSimple { seller_pct } => {
                out.insert("seller_pct".into(), json!(seller_pct));
            }
            MarginModel::Table { per_unit, default } => {
                let units: Map<String, Value> =
                    per_unit.iter().map(|(k, v)| (k.clone(), json!(v))).collect();
                out.insert("per_unit".into(), Value::Object(units));
                out.insert(
                    "default".into(),
                    default.as_ref().map_or(Value::Null, |d| d.name().into()),
                );
            }
            MarginModel::ByType(models) => {
                for (k, v) in models {
                    out.insert(k.as_str().to_string(), v.name().into());
                }
            }
            MarginModel::ByProduct { intraday, carry } => {
                out.insert("intraday".into(), intraday.describe_value());
                out.insert("carry".into(), carry.describe_value());
            }
        }
        Value::Object(out)
    }

    /// Name and parameters, nested models in full. What a result says the margin was.
    pub fn describe_value(&self) -> Value {
        let params = match self {
            MarginModel::ByType(models) => Value::Object(
                models.iter().map(|(k, v)| (k.as_str().to_string(), v.describe_value())).collect(),
            ),
            MarginModel::Table { default, .. } => {
                let mut p = match self.params() {
                    Value::Object(m) => m,
                    _ => Map::new(),
                };
                p.insert(
                    "default".into(),
                    default.as_ref().map_or(Value::Null, |d| d.describe_value()),
                );
                Value::Object(p)
            }
            _ => self.params(),
        };
        json!({"name": self.name(), "params": params})
    }

    /// [`MarginModel::describe_value`] as JSON text, keys sorted.
    pub fn describe(&self) -> String {
        self.describe_value().to_string()
    }
}

/// `FromSpec` over a set of option lines: sold legs that carry the
/// span-and-exposure deposit are margined as one position group and share its
/// requirement in proportion to what each would carry alone; every other
/// line keeps its own figure.
fn from_spec_shares(model: &MarginModel, legs: &[LineView<'_>]) -> Result<Vec<f64>> {
    let mut shares: Vec<f64> = legs.iter().map(|l| model.required(l)).collect::<Result<_>>()?;
    let mut grouped: Vec<(usize, OptionLeg)> = Vec::new();
    for (i, l) in legs.iter().enumerate() {
        let spec = l.instrument;
        let InstrumentKind::Option { strike, right, .. } = spec.kind else { continue };
        if spec.short_option_margin_per_contract().is_none() {
            continue;
        }
        grouped.push((
            i,
            OptionLeg {
                position_id: grouped.len() as u64,
                kernel: 0,
                strike,
                right,
                direction: if l.quantity < 0.0 { Direction::Short } else { Direction::Long },
                size: l.quantity.abs(),
                entry_price: l.price,
                multiplier: spec.multiplier,
                span_pct: spec.span_pct,
                exposure_pct: spec.exposure_pct,
            },
        ));
    }
    let option_legs: Vec<OptionLeg> = grouped.iter().map(|(_, l)| l.clone()).collect();
    let Some(requirement) = group_requirement(&option_legs) else { return Ok(shares) };
    for (_, position_id, share) in apportion(&option_legs, requirement.total) {
        let line = grouped[position_id as usize].0;
        shares[line] = share;
    }
    Ok(shares)
}

/// `OptionSimple`: a buyer pays the premium; a seller posts `seller_pct` of
/// the strike notional plus the premium.
fn option_simple(line: &LineView<'_>, seller_pct: f64) -> f64 {
    let premium = line.notional();
    if line.quantity > 0.0 {
        premium
    } else {
        let strike = match line.instrument.kind {
            InstrumentKind::Option { strike, .. } => strike,
            _ => line.price,
        };
        (strike * line.quantity * line.instrument.multiplier).abs() * seller_pct / 100.0 + premium
    }
}

/// What each option line of one underlying locks under `SpreadCredit`, in
/// the order of `legs`: its `OptionSimple` figure, less the credits below,
/// which come off the short lines they cover (a long always pays its
/// premium).
///
/// (b) Defined risk within one expiry. The lines of one expiry are priced
/// by [`defined_risk`], their worst loss at expiry. When that is below the
/// sum of the lines' own figures the difference is credited (at most the
/// shorts' figures), so the structure locks no less than its max loss. A
/// short with no covering long keeps a worst loss above its own figure, so
/// no credit. An expiry that earned this credit is not used again in (c).
///
/// (c) Calendar and diagonal. A short option is covered by a long option of
/// the same right and the same underlying that expires later. Source: NSE
/// Clearing's SPAN margin nets such offsetting positions across expiries of
/// one underlying (calendar spread benefit); the exchange's figure comes
/// from its scenario arrays, which a backtest does not have, so this is a
/// stated, conservative stand-in. The short's `seller_pct` margin is the
/// price move it must survive (`stress` = `seller_pct` of its strike). At
/// the short's expiry the long is worth at least its intrinsic value, so at
/// that move (from the strike as the spot proxy, as `OptionSimple` does) it
/// gains at least `intrinsic - its mark`, the time value left out. That
/// gain, up to `stress`, is credited per covered unit, but never so far
/// that the short keeps less than `spread_pct` of its strike notional (the
/// spread charge floor). Shorts are covered in order of expiry, by the
/// nearest later expiry first. `spread_pct` is NSE Clearing's published
/// calendar spread charge, a percentage of the far-month contract: 1.75 for
/// index derivatives, 2.2 for stock derivatives (SPAN Risk Parameters,
/// nseclearing.in/risk-management/equity-derivatives/span-risk-parameters,
/// page updated 20/06/2025, read 2026-10-21). That charge is only the spread
/// component of SPAN: the scanning risk left on the spread and the extreme
/// loss margin are not in it, so the figure is a floor, not the exchange's
/// whole margin.
///
/// (d) Expiry day. SEBI removed the calendar spread benefit on the expiry
/// day for the contracts expiring that day, for equity index derivatives
/// (circular SEBI/HO/MRD/TPD-1/P/CIR/2024/132 of 1 Oct 2024, clause 5.2.4:
/// effective 1 Feb 2025; NSE Clearing's FAQ on risk management gives 10 Feb
/// 2025, the date it was applied from). From the IST day of the short's
/// expiry its (c) credit is zero. Applied to the whole backtest range, before
/// and after that date, and to stock options too (whose treatment runs to
/// expiry): the stricter reading. (b) is not affected.
fn spread_credit_shares(
    legs: &[LineView<'_>],
    seller_pct: f64,
    spread_pct: f64,
    now: Timestamp,
) -> Result<Vec<f64>> {
    let mut shares: Vec<f64> = legs.iter().map(|l| option_simple(l, seller_pct)).collect();
    let option = |l: &LineView<'_>| match l.instrument.kind {
        InstrumentKind::Option { strike, right, .. } => Ok((strike, right)),
        _ => Err(invalid("spread_credit needs option legs with strikes".to_string())),
    };
    for l in legs {
        option(l)?;
    }
    let mut spent = vec![false; legs.len()];

    // (b)
    let mut by_expiry: BTreeMap<Option<Timestamp>, Vec<usize>> = BTreeMap::new();
    for (i, l) in legs.iter().enumerate() {
        by_expiry.entry(l.instrument.expiration_ns).or_default().push(i);
    }
    for idx in by_expiry.values().filter(|v| v.len() > 1) {
        let views: Vec<LineView<'_>> = idx.iter().map(|&i| legs[i]).collect();
        let worst = defined_risk(&views)?;
        let sum: f64 = idx.iter().map(|&i| shares[i]).sum();
        let shorts: f64 = idx.iter().filter(|&&i| legs[i].quantity < 0.0).map(|&i| shares[i]).sum();
        let credit = (sum - worst).min(shorts);
        if credit > 0.0 {
            for &i in idx {
                if legs[i].quantity < 0.0 {
                    shares[i] -= credit * shares[i] / shorts;
                }
                spent[i] = true;
            }
        }
    }

    // (c), (d)
    let expiry = |i: usize| legs[i].instrument.expiration_ns;
    let mut shorts: Vec<usize> = (0..legs.len())
        .filter(|&i| {
            legs[i].quantity < 0.0
                && !spent[i]
                && expiry(i).is_some_and(|e| {
                    let tz = zone_of(legs[i].instrument);
                    local_day(e, tz) > local_day(now, tz)
                })
        })
        .collect();
    shorts.sort_by_key(|&i| (expiry(i), i));
    let mut longs: Vec<usize> = (0..legs.len())
        .filter(|&i| legs[i].quantity > 0.0 && !spent[i] && expiry(i).is_some())
        .collect();
    longs.sort_by_key(|&i| (expiry(i), i));
    let mut left: Vec<f64> = legs.iter().map(|l| l.quantity.max(0.0)).collect();
    for s in shorts {
        let (k_s, right) = option(&legs[s])?;
        let stress = k_s * seller_pct / 100.0;
        let room = (stress - k_s * spread_pct / 100.0).max(0.0);
        let spot = match right {
            OptionRight::Call => k_s + stress,
            OptionRight::Put => k_s - stress,
        };
        let mut need = -legs[s].quantity;
        for &l in &longs {
            if need <= 0.0 {
                break;
            }
            let (k_l, right_l) = option(&legs[l])?;
            if right_l != right || expiry(l) <= expiry(s) || left[l] <= 0.0 {
                continue;
            }
            let intrinsic = match right {
                OptionRight::Call => (spot - k_l).max(0.0),
                OptionRight::Put => (k_l - spot).max(0.0),
            };
            let gain = (intrinsic - legs[l].price).max(0.0);
            let take = need.min(left[l]);
            let credit = take * legs[s].instrument.multiplier * gain.min(room);
            shares[s] -= credit.min(shares[s]);
            left[l] -= take;
            need -= take;
        }
    }
    Ok(shares)
}

/// The maximum loss of a set of legs at the option expiry, as the margin a
/// defined-risk position needs.
///
/// The legs' payoff against the underlying's price `S >= 0` at that instant
/// is piecewise linear: each option leg is `(intrinsic(S) - price)` per unit
/// and each future or cash leg is `(S - price)` per unit, times its signed
/// quantity and multiplier. So the worst loss is exact on the kinks (`S = 0`
/// and every strike) plus the slope beyond the highest strike: a position
/// that keeps losing as `S` rises (more short calls and short underlying than
/// long) has no bound and needs an infinite margin, which the account then
/// refuses. All option legs must share one expiry (the payoff of a longer
/// option at an earlier expiry is a pricing question, not intrinsic value:
/// use `SpreadCredit` for calendars); the future and cash legs are marked at
/// their price whatever their own expiry. Binary options have a stepped
/// payoff and are refused.
pub fn defined_risk(legs: &[LineView<'_>]) -> Result<f64> {
    let mut strikes: Vec<f64> = Vec::new();
    let mut expiry: Option<Option<Timestamp>> = None;
    for leg in legs {
        if let InstrumentKind::Option { strike, binary, .. } = leg.instrument.kind {
            if binary {
                return Err(invalid(format!(
                    "{}: defined_risk does not price a binary option's stepped payoff",
                    leg.instrument.symbol
                )));
            }
            match expiry {
                None => expiry = Some(leg.instrument.expiration_ns),
                Some(first) if first != leg.instrument.expiration_ns => {
                    return Err(invalid(format!(
                        "{}: defined_risk needs option legs of one expiry; margin a calendar \
                         with spread_credit",
                        leg.instrument.symbol
                    )))
                }
                Some(_) => {}
            }
            strikes.push(strike);
        }
    }
    if strikes.is_empty() {
        return Err(invalid("defined_risk needs option legs with strikes".to_string()));
    }
    strikes.sort_by(f64::total_cmp);
    strikes.dedup();

    // Payoff of the whole set if the underlying settles at `s`.
    let payoff = |s: f64| -> f64 {
        legs.iter()
            .map(|leg| {
                let units = leg.quantity * leg.instrument.multiplier;
                match leg.instrument.kind {
                    InstrumentKind::Option { strike: k, right, .. } => {
                        let intrinsic = match right {
                            OptionRight::Call => (s - k).max(0.0),
                            OptionRight::Put => (k - s).max(0.0),
                        };
                        (intrinsic - leg.price) * units
                    }
                    InstrumentKind::Index => 0.0,
                    _ => (s - leg.price) * units,
                }
            })
            .sum()
    };
    // Beyond the highest strike only calls and the linear legs still move.
    let slope: f64 = legs
        .iter()
        .filter(|leg| {
            !matches!(
                leg.instrument.kind,
                InstrumentKind::Option { right: OptionRight::Put, .. } | InstrumentKind::Index
            )
        })
        .map(|leg| leg.quantity * leg.instrument.multiplier)
        .sum();
    let scale: f64 = legs.iter().map(|leg| (leg.quantity * leg.instrument.multiplier).abs()).sum();
    if slope < -1e-12 * scale.max(1.0) {
        return Ok(f64::INFINITY);
    }
    let mut worst = 0.0f64;
    for s in std::iter::once(0.0).chain(strikes.iter().copied()) {
        worst = worst.min(payoff(s));
    }
    Ok(0.0 - worst)
}

#[cfg(test)]
#[path = "margin_model_tests.rs"]
mod tests;
