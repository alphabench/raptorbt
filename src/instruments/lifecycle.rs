//! Contract lifecycle terms and corporate actions, as data.
//!
//! Expiry and settlement are declared on the instrument ([`SettlementTerms`]):
//! how the contract settles (cash or physical delivery) and where its
//! settlement price comes from. Nothing is guessed: the price a contract
//! settled at, and which source gave it, travel on the settlement event.
//!
//! Corporate actions ([`CorporateAction`]) are data a run is given, never
//! inferred from prices: a split or bonus rescales the position held and a
//! cash dividend credits the account. See `docs/architecture/lifecycle.md`.

use serde::{Deserialize, Serialize};

use crate::core::error::{RaptorError, Result};
use crate::core::types::{Price, Timestamp};

/// How an expired contract is discharged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SettlementStyle {
    /// The position is closed at the settlement price and the difference is
    /// paid in cash. Nothing is delivered.
    #[default]
    Cash,
    /// The underlying is delivered. An option that finishes in the money is
    /// exercised (long) or assigned (short): the option line ends at nothing
    /// and the underlying is booked at the strike. A future is closed at its
    /// settlement price and the underlying is booked at that price.
    Physical,
}

impl SettlementStyle {
    pub fn as_str(self) -> &'static str {
        match self {
            SettlementStyle::Cash => "cash",
            SettlementStyle::Physical => "physical",
        }
    }

    pub fn parse(text: &str) -> Result<Self> {
        match text {
            "cash" => Ok(SettlementStyle::Cash),
            "physical" => Ok(SettlementStyle::Physical),
            other => Err(RaptorError::invalid_parameter(format!(
                "unknown settlement style {other:?}: \"cash\" or \"physical\""
            ))),
        }
    }
}

/// Where a contract's settlement price comes from.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub enum SettlementPrice {
    /// An option settles to intrinsic value when the underlying's price has
    /// been supplied, and at the contract's own last price when it has not;
    /// every other contract settles at its last price. The source used is
    /// reported on the settlement event, so the choice is never silent.
    #[default]
    Auto,
    /// The contract's own last price: the close of the bar at or after the
    /// expiry, or the last price seen when the expiry passes with no bar of
    /// the contract's own.
    LastPrice,
    /// The underlying's price, which must have been supplied: an option
    /// settles to intrinsic value against it, a future at the price itself.
    /// Without one the settlement is refused and the position stays open.
    Underlying,
    /// A price declared for this contract (the exchange's final settlement
    /// price). For a physically settled option it is the underlying's final
    /// price, which decides exercise; the option itself settles to nothing.
    Fixed(Price),
}

/// What gave a settlement its price.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SettlementSource {
    /// A price declared for the contract ([`SettlementPrice::Fixed`]).
    Declared,
    /// The underlying's price, supplied to the run.
    Underlying,
    /// The contract's own last price.
    LastPrice,
}

impl SettlementSource {
    pub fn as_str(self) -> &'static str {
        match self {
            SettlementSource::Declared => "declared",
            SettlementSource::Underlying => "underlying",
            SettlementSource::LastPrice => "last_price",
        }
    }
}

/// How an instrument settles at its expiry.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct SettlementTerms {
    pub style: SettlementStyle,
    pub price: SettlementPrice,
}

/// What a corporate action does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CorporateKind {
    /// `ratio` new units per old unit (a 1:2 split is 2).
    Split,
    /// `ratio` new units per old unit (a 1:1 bonus is 2).
    Bonus,
    /// A cash dividend of `amount` per unit held.
    Dividend,
}

impl CorporateKind {
    pub fn as_str(self) -> &'static str {
        match self {
            CorporateKind::Split => "split",
            CorporateKind::Bonus => "bonus",
            CorporateKind::Dividend => "dividend",
        }
    }

    pub fn parse(text: &str) -> Result<Self> {
        match text {
            "split" => Ok(CorporateKind::Split),
            "bonus" => Ok(CorporateKind::Bonus),
            "dividend" => Ok(CorporateKind::Dividend),
            other => Err(RaptorError::invalid_parameter(format!(
                "unknown corporate action kind {other:?}: \"split\", \"bonus\" or \"dividend\""
            ))),
        }
    }
}

/// A corporate action on one instrument, effective at `ts` (the ex-date's
/// first instant, in the run's nanosecond clock).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorporateAction {
    /// The instrument's symbol, as registered with the run.
    pub instrument: String,
    pub kind: CorporateKind,
    /// The ex-date: the instant from which the instrument trades on its new
    /// basis, and the instant at which a dividend's entitlement is fixed.
    pub ts: Timestamp,
    /// Split or bonus: new units per old unit. Unused for a dividend.
    pub ratio: f64,
    /// Dividend: cash per unit held, in the instrument's currency.
    pub amount: f64,
    /// Dividend: when the cash is credited. `None` credits it at `ts`.
    pub pay_ts: Option<Timestamp>,
}

impl CorporateAction {
    pub fn split(instrument: impl Into<String>, ts: Timestamp, ratio: f64) -> Self {
        Self {
            instrument: instrument.into(),
            kind: CorporateKind::Split,
            ts,
            ratio,
            amount: 0.0,
            pay_ts: None,
        }
    }

    pub fn bonus(instrument: impl Into<String>, ts: Timestamp, ratio: f64) -> Self {
        Self { kind: CorporateKind::Bonus, ..Self::split(instrument, ts, ratio) }
    }

    pub fn dividend(
        instrument: impl Into<String>,
        ts: Timestamp,
        amount: f64,
        pay_ts: Option<Timestamp>,
    ) -> Self {
        Self {
            instrument: instrument.into(),
            kind: CorporateKind::Dividend,
            ts,
            ratio: 0.0,
            amount,
            pay_ts,
        }
    }

    /// Whether this action rescales prices and quantities.
    pub fn rescales(&self) -> bool {
        matches!(self.kind, CorporateKind::Split | CorporateKind::Bonus)
    }

    /// Refuse a malformed action: a ratio that is not a positive finite
    /// number, a negative or non-finite dividend, a pay date before the
    /// ex-date.
    pub fn validate(&self) -> Result<()> {
        if self.instrument.is_empty() {
            return Err(RaptorError::invalid_parameter("a corporate action needs an instrument"));
        }
        match self.kind {
            CorporateKind::Split | CorporateKind::Bonus => {
                if !(self.ratio.is_finite() && self.ratio > 0.0) {
                    return Err(RaptorError::invalid_parameter(format!(
                        "{} on {}: ratio must be a positive finite number, got {}",
                        self.kind.as_str(),
                        self.instrument,
                        self.ratio
                    )));
                }
            }
            CorporateKind::Dividend => {
                if !(self.amount.is_finite() && self.amount >= 0.0) {
                    return Err(RaptorError::invalid_parameter(format!(
                        "dividend on {}: amount must be a finite number >= 0, got {}",
                        self.instrument, self.amount
                    )));
                }
                if self.pay_ts.is_some_and(|p| p < self.ts) {
                    return Err(RaptorError::invalid_parameter(format!(
                        "dividend on {}: pay date is before the ex-date",
                        self.instrument
                    )));
                }
            }
        }
        Ok(())
    }
}

/// How history reaches a strategy when corporate actions are declared.
///
/// Part of an indicator's identity: two strategies whose bars sit on
/// different bases never share an indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum AdjustmentPolicy {
    /// No corporate actions declared: prices are exactly as given.
    #[default]
    AsGiven,
    /// Prices are as they traded. A split shows as a jump in the series; the
    /// engine rescales the position at the ex-date, so cash stays exact.
    Raw,
    /// History before each split or bonus is rescaled onto the basis after
    /// it, so the series is continuous. The run trades on that one basis:
    /// the engine does not rescale the position again, and a cash dividend's
    /// per-unit amount is restated to the same basis.
    Adjusted,
}

impl AdjustmentPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            AdjustmentPolicy::AsGiven => "as_given",
            AdjustmentPolicy::Raw => "raw",
            AdjustmentPolicy::Adjusted => "adjusted",
        }
    }

    pub fn parse(text: &str) -> Result<Self> {
        match text {
            "as_given" => Ok(AdjustmentPolicy::AsGiven),
            "raw" => Ok(AdjustmentPolicy::Raw),
            "adjusted" => Ok(AdjustmentPolicy::Adjusted),
            other => Err(RaptorError::invalid_parameter(format!(
                "unknown adjustment policy {other:?}: \"raw\" or \"adjusted\""
            ))),
        }
    }
}

/// The factor that puts a price at `ts` onto the basis after every split and
/// bonus of `instrument` in `actions` that takes effect after `ts`: the
/// product of those ratios. A price on the adjusted basis is the raw price
/// divided by it, a volume is multiplied by it.
pub fn adjustment_factor(actions: &[CorporateAction], instrument: &str, ts: Timestamp) -> f64 {
    actions
        .iter()
        .filter(|a| a.rescales() && a.instrument == instrument && a.ts > ts)
        .fold(1.0, |f, a| f * a.ratio)
}

/// A dividend's per-unit amount on the adjusted basis: the amount divided by
/// the splits and bonuses of the same instrument that take effect after its
/// ex-date.
pub fn adjusted_dividend(actions: &[CorporateAction], action: &CorporateAction) -> f64 {
    action.amount / adjustment_factor(actions, &action.instrument, action.ts)
}

/// Put one instrument's bars on the adjusted basis (see
/// [`AdjustmentPolicy::Adjusted`]), in place. Prices of bars before an
/// action divide by the product of the ratios after them; volume multiplies.
/// `actions` of other instruments are ignored. Deterministic and idempotent
/// only in the sense that it must be applied once: the caller owns that.
#[allow(clippy::too_many_arguments)]
pub fn adjust_bars(
    instrument: &str,
    actions: &[CorporateAction],
    timestamps: &[Timestamp],
    open: &mut [f64],
    high: &mut [f64],
    low: &mut [f64],
    close: &mut [f64],
    volume: &mut [f64],
) -> Result<()> {
    let n = timestamps.len();
    if [open.len(), high.len(), low.len(), close.len(), volume.len()].iter().any(|&l| l != n) {
        return Err(RaptorError::invalid_parameter(
            "adjust_bars: every column must have one entry per timestamp",
        ));
    }
    let mut own: Vec<&CorporateAction> =
        actions.iter().filter(|a| a.rescales() && a.instrument == instrument).collect();
    own.sort_by_key(|a| a.ts);
    if own.is_empty() {
        return Ok(());
    }
    for i in 0..n {
        let f = own.iter().filter(|a| a.ts > timestamps[i]).fold(1.0, |f, a| f * a.ratio);
        if f != 1.0 {
            open[i] /= f;
            high[i] /= f;
            low[i] /= f;
            close[i] /= f;
            volume[i] *= f;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_ratio_must_be_positive_and_finite() {
        assert!(CorporateAction::split("A", 1, 2.0).validate().is_ok());
        assert!(CorporateAction::split("A", 1, 0.0).validate().is_err());
        assert!(CorporateAction::bonus("A", 1, f64::NAN).validate().is_err());
        assert!(CorporateAction::dividend("A", 5, -1.0, None).validate().is_err());
        assert!(CorporateAction::dividend("A", 5, 1.0, Some(4)).validate().is_err());
        assert!(CorporateAction::dividend("A", 5, 1.0, Some(9)).validate().is_ok());
    }

    #[test]
    fn history_divides_by_every_later_ratio() {
        let acts = vec![CorporateAction::split("A", 20, 2.0), CorporateAction::bonus("A", 40, 1.5)];
        assert_eq!(adjustment_factor(&acts, "A", 10), 3.0);
        assert_eq!(adjustment_factor(&acts, "A", 20), 1.5);
        assert_eq!(adjustment_factor(&acts, "A", 40), 1.0);
        assert_eq!(adjustment_factor(&acts, "B", 10), 1.0);
        let ts = [10, 20, 40];
        let (mut o, mut h, mut l, mut c) = ([60.0; 3], [60.0; 3], [60.0; 3], [60.0; 3]);
        let mut v = [100.0; 3];
        adjust_bars("A", &acts, &ts, &mut o, &mut h, &mut l, &mut c, &mut v).unwrap();
        assert_eq!(c, [20.0, 40.0, 60.0]);
        assert_eq!(v, [300.0, 150.0, 100.0]);
    }

    #[test]
    fn a_dividend_is_restated_to_the_adjusted_basis() {
        let acts = vec![
            CorporateAction::dividend("A", 10, 6.0, None),
            CorporateAction::split("A", 20, 2.0),
        ];
        assert_eq!(adjusted_dividend(&acts, &acts[0]), 3.0);
    }
}
