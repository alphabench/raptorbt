//! Option chain: pick contracts by expiry and strike, with no lookahead.
//!
//! An [`OptionChain`] is built once from instrument specs. Every query takes
//! the current event time `now` (nanoseconds), and answers only from
//! contracts listed at that time: a contract is visible from its
//! `activation_ns` (always, when it has none) up to but not including its
//! `expiration_ns`. A contract that is not yet listed cannot be selected, and
//! neither can its strike or expiry appear in any list.
//!
//! Selection is deterministic: lists are ascending, and when two strikes are
//! equally near a target the lower strike wins.
//!
//! One thing is deliberately computed from the whole chain: whether an
//! expiry is a month's last ("monthly"). That is a calendar fact about the
//! expiry, so it does not change as contracts get listed; build the chain
//! from the full expiry calendar if you select by `Monthly` or `Weekly`.

use std::collections::BTreeMap;

use thiserror::Error;

use super::{greeks, Model, OptionInputs};
use crate::core::types::Timestamp;
use crate::instruments::{InstrumentKind, InstrumentSpec, OptionRight};

const DAY_NS: i64 = 86_400_000_000_000;
/// Nanoseconds in a 365-day year: the year `tau` is measured in.
pub const YEAR_NS: i64 = 365 * DAY_NS;

/// Why a chain could not be built.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ChainError {
    #[error("{0}")]
    Invalid(String),
}

/// Which expiry to pick among those listed at `now`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpiryPick {
    /// The first expiry at or after `now + min_days` days.
    Nearest { min_days: i64 },
    /// The `n`th listed expiry, 0 being the nearest.
    Nth(usize),
    /// The `n`th listed expiry that is not the last of its calendar month.
    Weekly(usize),
    /// The `n`th listed expiry that is the last of its calendar month.
    Monthly(usize),
    /// The listed expiry on the same calendar day as this timestamp.
    Date(Timestamp),
}

fn strike_key(strike: f64) -> u64 {
    // Strikes are positive and finite, so the IEEE bits order like the values.
    strike.to_bits()
}

#[derive(Debug, Clone, Default)]
struct StrikeSlot {
    strike: f64,
    call: Option<usize>,
    put: Option<usize>,
}

#[derive(Debug, Clone, Default)]
struct UnderlyingChain {
    expiries: BTreeMap<Timestamp, BTreeMap<u64, StrikeSlot>>,
}

/// Option contracts grouped by underlying, expiry, strike and right.
#[derive(Debug, Clone, Default)]
pub struct OptionChain {
    specs: Vec<InstrumentSpec>,
    by_underlying: BTreeMap<String, UnderlyingChain>,
}

/// Days since 1970-01-01 to (year, month).
fn year_month(days: i64) -> (i64, u32) {
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m)
}

fn local_day(ts: Timestamp, utc_offset_ns: i64) -> i64 {
    ts.saturating_add(utc_offset_ns).div_euclid(DAY_NS)
}

impl OptionChain {
    /// Build a chain from specs. Specs that are not options are ignored. An
    /// option needs an underlying, an expiry and a finite positive strike; two
    /// contracts with the same underlying, expiry, strike and right are
    /// refused.
    pub fn new(specs: impl IntoIterator<Item = InstrumentSpec>) -> Result<Self, ChainError> {
        let mut chain = OptionChain::default();
        for spec in specs {
            let InstrumentKind::Option { strike, right, underlying, .. } = &spec.kind else {
                continue;
            };
            let sym = spec.symbol.clone();
            let invalid = |m: &str| Err(ChainError::Invalid(format!("{sym}: {m}")));
            let Some(underlying) = underlying.clone() else {
                return invalid("option has no underlying");
            };
            let Some(expiry) = spec.expiration_ns else {
                return invalid("option has no expiration_ns");
            };
            if !strike.is_finite() || *strike <= 0.0 {
                return invalid("strike must be finite and positive");
            }
            let (strike, right) = (*strike, *right);
            let idx = chain.specs.len();
            let slot = chain
                .by_underlying
                .entry(underlying.clone())
                .or_default()
                .expiries
                .entry(expiry)
                .or_default()
                .entry(strike_key(strike))
                .or_insert_with(|| StrikeSlot { strike, ..Default::default() });
            let place = match right {
                OptionRight::Call => &mut slot.call,
                OptionRight::Put => &mut slot.put,
            };
            if let Some(other) = *place {
                return Err(ChainError::Invalid(format!(
                    "two contracts for {underlying} expiry {expiry} strike {strike} {right:?}: \
                     {}, {}",
                    chain.specs[other].symbol, spec.symbol
                )));
            }
            *place = Some(idx);
            chain.specs.push(spec);
        }
        Ok(chain)
    }

    pub fn underlyings(&self) -> impl Iterator<Item = &str> + '_ {
        self.by_underlying.keys().map(String::as_str)
    }

    /// Number of option contracts held (listed or not).
    pub fn len(&self) -> usize {
        self.specs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }

    fn live(&self, idx: Option<usize>, now: Timestamp) -> Option<&InstrumentSpec> {
        idx.map(|i| &self.specs[i]).filter(|s| s.is_live_at(now))
    }

    /// Whether a slot holds a listed contract of `right` (either, when `None`).
    fn slot_contract(&self, slot: &StrikeSlot, right: Option<OptionRight>, now: Timestamp) -> bool {
        let call = self.live(slot.call, now).is_some();
        let put = self.live(slot.put, now).is_some();
        match right {
            Some(OptionRight::Call) => call,
            Some(OptionRight::Put) => put,
            None => call || put,
        }
    }

    /// Expiries with at least one contract listed at `now`, ascending.
    pub fn expiries(&self, underlying: &str, now: Timestamp) -> Vec<Timestamp> {
        let Some(u) = self.by_underlying.get(underlying) else { return Vec::new() };
        u.expiries
            .iter()
            .filter(|(&e, strikes)| {
                e > now && strikes.values().any(|s| self.slot_contract(s, None, now))
            })
            .map(|(&e, _)| e)
            .collect()
    }

    /// Strikes with a listed contract at `now` (of `right`, when given),
    /// ascending.
    pub fn strikes(
        &self,
        underlying: &str,
        expiry: Timestamp,
        now: Timestamp,
        right: Option<OptionRight>,
    ) -> Vec<f64> {
        let Some(strikes) =
            self.by_underlying.get(underlying).and_then(|u| u.expiries.get(&expiry))
        else {
            return Vec::new();
        };
        strikes.values().filter(|s| self.slot_contract(s, right, now)).map(|s| s.strike).collect()
    }

    /// The contract at exactly these terms, if listed at `now`.
    pub fn contract(
        &self,
        underlying: &str,
        expiry: Timestamp,
        strike: f64,
        right: OptionRight,
        now: Timestamp,
    ) -> Option<&InstrumentSpec> {
        let slot =
            self.by_underlying.get(underlying)?.expiries.get(&expiry)?.get(&strike_key(strike))?;
        let idx = match right {
            OptionRight::Call => slot.call,
            OptionRight::Put => slot.put,
        };
        self.live(idx, now)
    }

    /// Contracts of one expiry listed at `now`, by ascending strike (calls
    /// before puts at the same strike), optionally of one right.
    pub fn contracts(
        &self,
        underlying: &str,
        expiry: Timestamp,
        now: Timestamp,
        right: Option<OptionRight>,
    ) -> Vec<&InstrumentSpec> {
        let Some(strikes) =
            self.by_underlying.get(underlying).and_then(|u| u.expiries.get(&expiry))
        else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for slot in strikes.values() {
            if right != Some(OptionRight::Put) {
                out.extend(self.live(slot.call, now));
            }
            if right != Some(OptionRight::Call) {
                out.extend(self.live(slot.put, now));
            }
        }
        out
    }

    /// Pick an expiry among those listed at `now`. `utc_offset_ns` sets the
    /// calendar used for month and day (0 = UTC).
    pub fn pick_expiry(
        &self,
        underlying: &str,
        now: Timestamp,
        pick: ExpiryPick,
        utc_offset_ns: i64,
    ) -> Option<Timestamp> {
        let listed = self.expiries(underlying, now);
        match pick {
            ExpiryPick::Nearest { min_days } => {
                let from = now.saturating_add(min_days.saturating_mul(DAY_NS));
                listed.into_iter().find(|&e| e >= from)
            }
            ExpiryPick::Nth(n) => listed.get(n).copied(),
            ExpiryPick::Date(ts) => {
                let day = local_day(ts, utc_offset_ns);
                listed.into_iter().find(|&e| local_day(e, utc_offset_ns) == day)
            }
            ExpiryPick::Weekly(n) | ExpiryPick::Monthly(n) => {
                let monthly = matches!(pick, ExpiryPick::Monthly(_));
                let all = &self.by_underlying.get(underlying)?.expiries;
                // The last expiry of each calendar month, over the whole chain.
                let mut last: BTreeMap<(i64, u32), Timestamp> = BTreeMap::new();
                for &e in all.keys() {
                    last.insert(year_month(local_day(e, utc_offset_ns)), e);
                }
                listed
                    .into_iter()
                    .filter(|&e| {
                        let is_monthly =
                            last.get(&year_month(local_day(e, utc_offset_ns))) == Some(&e);
                        is_monthly == monthly
                    })
                    .nth(n)
            }
        }
    }

    /// The listed strike nearest `reference` (the lower on a tie).
    pub fn atm_strike(
        &self,
        underlying: &str,
        expiry: Timestamp,
        now: Timestamp,
        reference: f64,
    ) -> Option<f64> {
        if !reference.is_finite() {
            return None;
        }
        let mut best: Option<(f64, f64)> = None;
        for k in self.strikes(underlying, expiry, now, None) {
            let d = (k - reference).abs();
            if best.is_none_or(|(_, bd)| d < bd) {
                best = Some((k, d));
            }
        }
        best.map(|(k, _)| k)
    }

    /// The strike `offset` listed strikes above the one nearest `reference`
    /// (negative: below). `None` when it falls off the chain.
    pub fn strike_at_offset(
        &self,
        underlying: &str,
        expiry: Timestamp,
        now: Timestamp,
        reference: f64,
        offset: i64,
    ) -> Option<f64> {
        let strikes = self.strikes(underlying, expiry, now, None);
        let atm = self.atm_strike(underlying, expiry, now, reference)?;
        let i = strikes.iter().position(|&k| k == atm)? as i64;
        usize::try_from(i.checked_add(offset)?).ok().and_then(|j| strikes.get(j).copied())
    }

    /// The listed contract of `right` whose `measure` is nearest `target`
    /// (the lower strike on a tie). Contracts for which `measure` answers
    /// `None` or NaN are skipped.
    pub fn closest_by(
        &self,
        underlying: &str,
        expiry: Timestamp,
        now: Timestamp,
        right: OptionRight,
        target: f64,
        mut measure: impl FnMut(&InstrumentSpec) -> Option<f64>,
    ) -> Option<&InstrumentSpec> {
        let mut best: Option<(&InstrumentSpec, f64)> = None;
        for c in self.contracts(underlying, expiry, now, Some(right)) {
            let Some(m) = measure(c).filter(|m| !m.is_nan()) else { continue };
            let d = (m - target).abs();
            if best.is_none_or(|(_, bd)| d < bd) {
                best = Some((c, d));
            }
        }
        best.map(|(c, _)| c)
    }

    /// The listed contract of `right` whose model delta is nearest `target`
    /// (calls positive, puts negative), at a flat `vol`. `underlying_price`
    /// is the spot for [`Model::BlackScholes`], the forward for
    /// [`Model::Black76`]. Time to expiry is measured from `now`.
    #[allow(clippy::too_many_arguments)]
    pub fn by_delta(
        &self,
        underlying: &str,
        expiry: Timestamp,
        now: Timestamp,
        right: OptionRight,
        target: f64,
        model: Model,
        underlying_price: f64,
        vol: f64,
        rate: f64,
        dividend_yield: f64,
    ) -> Option<&InstrumentSpec> {
        let tau = (expiry - now) as f64 / YEAR_NS as f64;
        self.closest_by(underlying, expiry, now, right, target, |c| {
            let InstrumentKind::Option { strike, .. } = &c.kind else { return None };
            let inputs = OptionInputs {
                right,
                underlying: underlying_price,
                strike: *strike,
                tau,
                vol,
                rate,
                dividend_yield,
            };
            greeks(model, &inputs).ok().map(|g| g.delta)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const H: i64 = 3_600_000_000_000;
    // 2026-01-01 00:00 UTC, a Thursday.
    const T0: i64 = 1_767_225_600_000_000_000;
    const U: &str = "IDX";

    fn opt(
        strike: f64,
        right: OptionRight,
        expiry: i64,
        activation: Option<i64>,
    ) -> InstrumentSpec {
        let tag = if right == OptionRight::Call { "C" } else { "P" };
        let mut s = InstrumentSpec::new(
            format!("{U}-{expiry}-{strike}-{tag}"),
            InstrumentKind::Option { strike, right, underlying: Some(U.into()), binary: false },
        );
        s.expiration_ns = Some(expiry);
        s.activation_ns = activation;
        s.lot_size = 50.0;
        s
    }

    /// Weekly expiries on Jan 8, 15, 22, 29 and Feb 5 (Thursdays); strikes
    /// 90..=110 step 5; calls and puts.
    fn chain() -> OptionChain {
        let mut specs = vec![InstrumentSpec::new(U, InstrumentKind::Index)];
        for w in 1..=5 {
            let e = T0 + 7 * w * DAY_NS;
            for k in (90..=110).step_by(5) {
                specs.push(opt(k as f64, OptionRight::Call, e, None));
                specs.push(opt(k as f64, OptionRight::Put, e, None));
            }
        }
        OptionChain::new(specs).unwrap()
    }

    #[test]
    fn expiries_are_listed_ascending_and_drop_when_expired() {
        let c = chain();
        let e = |w: i64| T0 + 7 * w * DAY_NS;
        assert_eq!(c.expiries(U, T0), [e(1), e(2), e(3), e(4), e(5)]);
        assert_eq!(c.expiries(U, e(1)), [e(2), e(3), e(4), e(5)], "expiry itself is gone");
        assert_eq!(c.expiries(U, e(1) - 1), [e(1), e(2), e(3), e(4), e(5)]);
        assert!(c.expiries("OTHER", T0).is_empty());
    }

    #[test]
    fn contracts_listed_later_are_invisible_until_their_time() {
        let e = T0 + 7 * DAY_NS;
        let listed_at = T0 + 3 * DAY_NS;
        let c = OptionChain::new([
            opt(100.0, OptionRight::Call, e, None),
            opt(105.0, OptionRight::Call, e, Some(listed_at)),
        ])
        .unwrap();
        assert_eq!(c.strikes(U, e, listed_at - 1, None), [100.0]);
        assert_eq!(c.strikes(U, e, listed_at, None), [100.0, 105.0]);
        assert!(c.contract(U, e, 105.0, OptionRight::Call, listed_at - 1).is_none());
        assert!(c.contract(U, e, 105.0, OptionRight::Call, listed_at).is_some());
        // The nearest strike to 106 is 100 until 105 is listed.
        assert_eq!(c.atm_strike(U, e, listed_at - 1, 106.0), Some(100.0));
        assert_eq!(c.atm_strike(U, e, listed_at, 106.0), Some(105.0));
        // An expiry none of whose contracts is listed is not an expiry yet.
        let later = T0 + 20 * DAY_NS;
        let c = OptionChain::new([opt(100.0, OptionRight::Put, later, Some(T0 + 10 * DAY_NS))])
            .unwrap();
        assert!(c.expiries(U, T0).is_empty());
        assert_eq!(c.expiries(U, T0 + 10 * DAY_NS), [later]);
    }

    #[test]
    fn expiry_picks() {
        let c = chain();
        let e = |w: i64| T0 + 7 * w * DAY_NS;
        let now = T0 + 2 * DAY_NS;
        let pick = |p| c.pick_expiry(U, now, p, 0);
        assert_eq!(pick(ExpiryPick::Nearest { min_days: 0 }), Some(e(1)));
        assert_eq!(pick(ExpiryPick::Nearest { min_days: 6 }), Some(e(2)));
        assert_eq!(pick(ExpiryPick::Nth(2)), Some(e(3)));
        assert_eq!(pick(ExpiryPick::Nth(9)), None);
        assert_eq!(pick(ExpiryPick::Date(e(4) + 5 * H)), Some(e(4)));
        assert_eq!(pick(ExpiryPick::Date(e(4) + DAY_NS)), None);
        // Jan 29 closes January; Feb 5 closes this chain's February.
        assert_eq!(pick(ExpiryPick::Monthly(0)), Some(e(4)));
        assert_eq!(pick(ExpiryPick::Monthly(1)), Some(e(5)));
        assert_eq!(pick(ExpiryPick::Weekly(0)), Some(e(1)));
        assert_eq!(pick(ExpiryPick::Weekly(2)), Some(e(3)));
        assert_eq!(pick(ExpiryPick::Weekly(3)), None);
    }

    #[test]
    fn monthly_does_not_change_as_time_passes() {
        let c = chain();
        let e4 = T0 + 28 * DAY_NS;
        assert_eq!(c.pick_expiry(U, T0, ExpiryPick::Monthly(0), 0), Some(e4));
        assert_eq!(c.pick_expiry(U, T0 + 21 * DAY_NS + H, ExpiryPick::Monthly(0), 0), Some(e4));
    }

    #[test]
    fn calendar_offset_moves_the_month_boundary() {
        // 2026-01-31 20:00 UTC is already February 1 at UTC+5:30.
        let e = T0 + 30 * DAY_NS + 20 * H;
        let c = OptionChain::new([
            opt(100.0, OptionRight::Call, e - 7 * DAY_NS, None),
            opt(100.0, OptionRight::Call, e, None),
        ])
        .unwrap();
        // UTC: both in January, the later one is the month's last.
        assert_eq!(c.pick_expiry(U, T0, ExpiryPick::Monthly(0), 0), Some(e));
        // UTC+5:30: the later one is February's only expiry; the earlier is
        // January's last.
        let ist = 5 * H + 30 * 60_000_000_000;
        assert_eq!(c.pick_expiry(U, T0, ExpiryPick::Monthly(0), ist), Some(e - 7 * DAY_NS));
        assert_eq!(c.pick_expiry(U, T0, ExpiryPick::Monthly(1), ist), Some(e));
    }

    #[test]
    fn atm_offsets_and_ties() {
        let c = chain();
        let e = T0 + 7 * DAY_NS;
        assert_eq!(c.atm_strike(U, e, T0, 101.0), Some(100.0));
        assert_eq!(c.atm_strike(U, e, T0, 102.5), Some(100.0), "tie goes to the lower strike");
        assert_eq!(c.atm_strike(U, e, T0, 102.6), Some(105.0));
        assert_eq!(c.atm_strike(U, e, T0, 500.0), Some(110.0));
        assert_eq!(c.strike_at_offset(U, e, T0, 101.0, 2), Some(110.0));
        assert_eq!(c.strike_at_offset(U, e, T0, 101.0, -2), Some(90.0));
        assert_eq!(c.strike_at_offset(U, e, T0, 101.0, 3), None);
        assert_eq!(c.strike_at_offset(U, e, T0, 101.0, -3), None);
        assert_eq!(c.atm_strike(U, e, T0, f64::NAN), None);
    }

    #[test]
    fn delta_and_premium_selection() {
        let c = chain();
        let e = T0 + 7 * DAY_NS;
        let strike = |s: Option<&InstrumentSpec>| match &s.unwrap().kind {
            InstrumentKind::Option { strike, .. } => *strike,
            _ => unreachable!(),
        };
        // Spot 100, 20% vol, one week: the 25-delta call is an upper strike,
        // the 25-delta put a lower one.
        let call = c.by_delta(
            U,
            e,
            T0,
            OptionRight::Call,
            0.25,
            Model::BlackScholes,
            100.0,
            0.5,
            0.0,
            0.0,
        );
        let put = c.by_delta(
            U,
            e,
            T0,
            OptionRight::Put,
            -0.25,
            Model::BlackScholes,
            100.0,
            0.5,
            0.0,
            0.0,
        );
        assert!(strike(call) > 100.0 && strike(put) < 100.0);
        // Premium from quotes: closest to 3.0 among the calls.
        let quotes = [(90.0, 10.5), (95.0, 6.0), (100.0, 3.2), (105.0, 1.0), (110.0, 0.2)];
        let by_quote = c.closest_by(U, e, T0, OptionRight::Call, 3.0, |s| {
            let InstrumentKind::Option { strike, .. } = &s.kind else { return None };
            quotes.iter().find(|(k, _)| k == strike).map(|&(_, p)| p)
        });
        assert_eq!(strike(by_quote), 100.0);
    }

    #[test]
    fn duplicates_and_incomplete_options_are_refused() {
        let e = T0 + DAY_NS;
        let a = opt(100.0, OptionRight::Call, e, None);
        let mut b = a.clone();
        b.symbol = "OTHER".into();
        assert!(OptionChain::new([a.clone(), b]).is_err());
        let mut no_exp = a.clone();
        no_exp.expiration_ns = None;
        assert!(OptionChain::new([no_exp]).is_err());
        let orphan = InstrumentSpec::new(
            "X",
            InstrumentKind::Option {
                strike: 1.0,
                right: OptionRight::Call,
                underlying: None,
                binary: false,
            },
        );
        assert!(OptionChain::new([orphan]).is_err());
        assert!(OptionChain::new(Vec::new()).unwrap().is_empty());
    }
}
