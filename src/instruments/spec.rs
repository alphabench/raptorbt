//! Instrument market definitions.
//!
//! [`InstrumentSpec`] describes *what is being traded* — tick size, lot size,
//! contract multiplier, expiry — as opposed to [`InstrumentConfig`], which
//! describes *how this run allocates to it* (capital cap, stop/target
//! overrides). The kernel consumes both: the spec drives price/size
//! quantization, notional scaling, and expiry settlement; the config retains
//! its existing role and its `lot_size` wins over the spec's when both are
//! set, since it is the user's explicit per-run override.
//!
//! [`InstrumentConfig`]: crate::core::types::InstrumentConfig

use serde::{Deserialize, Serialize};

use crate::core::types::{Direction, Price, Timestamp};

/// Which side an option contract is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptionRight {
    Call,
    Put,
}

/// What kind of instrument a spec describes.
///
/// One enum rather than a type hierarchy: the variants differ only in a few
/// fields, and a flat enum keeps the kernel `Copy`-friendly and the Python
/// binding surface small.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InstrumentKind {
    /// Cash instruments: equity, spot commodity, tokenized assets.
    Cash,
    /// A currency pair (spot FX).
    CurrencyPair,
    /// Linear derivative contracts: futures, CFDs, perpetuals.
    ///
    /// A contract with an expiration is a future; without one, a perpetual.
    Contract { underlying: Option<String> },
    /// Vanilla or binary option.
    Option { strike: Price, right: OptionRight, underlying: Option<String>, binary: bool },
    /// Non-tradable reference index.
    Index,
}

impl InstrumentKind {
    /// Whether orders may be placed on this instrument.
    #[inline]
    pub fn tradable(&self) -> bool {
        !matches!(self, InstrumentKind::Index)
    }
}

/// Market definition of one instrument.
///
/// All numeric fields default to values that reproduce the engine's behavior
/// without a spec: increments of `0.0` mean "unquantized", `multiplier` of
/// `1.0` means prices are already notional, and absent timestamps mean the
/// instrument is always live.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InstrumentSpec {
    pub symbol: String,
    pub kind: InstrumentKind,
    /// Minimum price step (tick size); `0.0` leaves prices unquantized.
    pub price_increment: f64,
    /// Minimum size step; `0.0` leaves sizes unquantized beyond lot rounding.
    pub size_increment: f64,
    /// Contract lot size; sizing floors to a whole number of lots.
    pub lot_size: f64,
    /// Contract point value: notional = price * size * multiplier.
    pub multiplier: f64,
    /// Initial margin as a fraction of notional; `0.0` = fully funded.
    ///
    /// Stored for the account layer; the kernel itself does not consume it.
    pub margin_init: f64,
    /// Maintenance margin as a fraction of notional.
    pub margin_maint: f64,
    /// Maker fee fraction, for fee models that distinguish liquidity roles.
    pub maker_fee: f64,
    /// Taker fee fraction.
    pub taker_fee: f64,
    /// First tradable timestamp (ns); entries before it are rejected.
    pub activation_ns: Option<Timestamp>,
    /// Expiry timestamp (ns); open positions settle at the first bar at or
    /// past it and later entries are rejected.
    pub expiration_ns: Option<Timestamp>,
}

impl InstrumentSpec {
    /// A spec with neutral defaults for the given symbol and kind.
    pub fn new(symbol: impl Into<String>, kind: InstrumentKind) -> Self {
        Self {
            symbol: symbol.into(),
            kind,
            price_increment: 0.0,
            size_increment: 0.0,
            lot_size: 1.0,
            multiplier: 1.0,
            margin_init: 0.0,
            margin_maint: 0.0,
            maker_fee: 0.0,
            taker_fee: 0.0,
            activation_ns: None,
            expiration_ns: None,
        }
    }

    /// Whether the instrument has expired at `ts`.
    #[inline]
    pub fn is_expired_at(&self, ts: Timestamp) -> bool {
        self.expiration_ns.is_some_and(|exp| ts >= exp)
    }

    /// Whether the instrument is tradable at `ts` (active and not expired).
    #[inline]
    pub fn is_live_at(&self, ts: Timestamp) -> bool {
        let activated = self.activation_ns.is_none_or(|act| ts >= act);
        activated && !self.is_expired_at(ts)
    }

    /// Quantize a derived protective price (stop or target) conservatively.
    ///
    /// Rounds toward the losing side for the given position direction — down
    /// for longs, up for shorts — so quantization never flatters the result:
    /// a long's stop moves further from entry (larger loss) and its target
    /// closer (smaller win), and symmetrically for shorts.
    pub fn quantize_protective(&self, price: Price, direction: Direction) -> Price {
        if self.price_increment <= 0.0 {
            return price;
        }
        let ticks = price / self.price_increment;
        // Snap to the nearest tick when the price is already on the grid up
        // to float noise (100.05 / 0.05 = 2000.9999...), so on-grid prices
        // never lose a whole tick to floor/ceil.
        let nearest = ticks.round();
        let rounded = if (ticks - nearest).abs() < 1e-9 * nearest.abs().max(1.0) {
            nearest
        } else {
            match direction {
                Direction::Long => ticks.floor(),
                Direction::Short => ticks.ceil(),
            }
        };
        rounded * self.price_increment
    }

    /// Quantize a position size down to the size increment.
    ///
    /// Applied after lot rounding; a zero increment is a no-op.
    pub fn quantize_size(&self, size: f64) -> f64 {
        if self.size_increment <= 0.0 {
            return size;
        }
        let steps = size / self.size_increment;
        let nearest = steps.round();
        let rounded = if (steps - nearest).abs() < 1e-9 * nearest.abs().max(1.0) {
            nearest
        } else {
            steps.floor()
        };
        rounded * self.size_increment
    }

    /// Intrinsic settlement value per unit at expiry.
    ///
    /// Options settle to intrinsic value against the underlying price when
    /// one is known; everything else (and options without one) settles at the
    /// provided fallback price, which callers pass as the settlement bar's
    /// close.
    pub fn settlement_value(&self, fallback_price: Price, underlying_price: Option<Price>) -> Price {
        match (&self.kind, underlying_price) {
            (InstrumentKind::Option { strike, right, binary, .. }, Some(under)) => {
                let intrinsic = match right {
                    OptionRight::Call => (under - strike).max(0.0),
                    OptionRight::Put => (strike - under).max(0.0),
                };
                if *binary {
                    if intrinsic > 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    intrinsic
                }
            }
            _ => fallback_price,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_neutral() {
        let spec = InstrumentSpec::new("X", InstrumentKind::Cash);
        assert_eq!(spec.multiplier, 1.0);
        assert_eq!(spec.lot_size, 1.0);
        assert_eq!(spec.quantize_protective(101.2345, Direction::Long), 101.2345);
        assert_eq!(spec.quantize_size(123.456), 123.456);
        assert!(spec.is_live_at(0));
        assert!(spec.is_live_at(i64::MAX));
    }

    #[test]
    fn protective_quantization_is_conservative() {
        let spec = InstrumentSpec {
            price_increment: 0.05,
            ..InstrumentSpec::new("X", InstrumentKind::Cash)
        };
        // Long: floor — stop drifts away from entry, target toward it.
        assert!((spec.quantize_protective(99.98, Direction::Long) - 99.95).abs() < 1e-9);
        // Short: ceil.
        assert!((spec.quantize_protective(100.02, Direction::Short) - 100.05).abs() < 1e-9);
        // Already on-tick prices are unchanged.
        assert!((spec.quantize_protective(100.05, Direction::Long) - 100.05).abs() < 1e-9);
    }

    #[test]
    fn size_quantization_floors() {
        let spec = InstrumentSpec {
            size_increment: 0.001,
            ..InstrumentSpec::new("X", InstrumentKind::CurrencyPair)
        };
        assert!((spec.quantize_size(1.23456) - 1.234).abs() < 1e-9);
    }

    #[test]
    fn expiry_windows() {
        let spec = InstrumentSpec {
            activation_ns: Some(100),
            expiration_ns: Some(200),
            ..InstrumentSpec::new("X", InstrumentKind::Contract { underlying: None })
        };
        assert!(!spec.is_live_at(99));
        assert!(spec.is_live_at(100));
        assert!(spec.is_live_at(199));
        assert!(!spec.is_live_at(200));
        assert!(spec.is_expired_at(200));
    }

    #[test]
    fn option_settles_to_intrinsic() {
        let call = InstrumentSpec::new(
            "C",
            InstrumentKind::Option {
                strike: 100.0,
                right: OptionRight::Call,
                underlying: None,
                binary: false,
            },
        );
        assert_eq!(call.settlement_value(7.0, Some(110.0)), 10.0);
        assert_eq!(call.settlement_value(7.0, Some(90.0)), 0.0);
        // No underlying price: fall back to the provided price.
        assert_eq!(call.settlement_value(7.0, None), 7.0);

        let binary_put = InstrumentSpec::new(
            "B",
            InstrumentKind::Option {
                strike: 100.0,
                right: OptionRight::Put,
                underlying: None,
                binary: true,
            },
        );
        assert_eq!(binary_put.settlement_value(0.4, Some(90.0)), 1.0);
        assert_eq!(binary_put.settlement_value(0.4, Some(110.0)), 0.0);
    }

    #[test]
    fn index_is_not_tradable() {
        assert!(!InstrumentKind::Index.tradable());
        assert!(InstrumentKind::Cash.tradable());
    }
}
