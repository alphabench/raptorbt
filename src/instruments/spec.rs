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

use crate::core::session_schedule::SessionSchedule;
use crate::core::types::{Direction, Price, Timestamp};
use crate::instruments::lifecycle::SettlementTerms;

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
    /// Contract lot size; sizing floors to a whole number of lots. `1.0` is
    /// the neutral default and floors nothing: whole units are
    /// `size_increment = 1.0`.
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
    /// Fee fraction charged on settlement at expiry, applied to the settled
    /// notional. `0.0` (default) settles free, as before.
    ///
    /// Separate from `taker_fee` because exchanges commonly price exercise
    /// and assignment differently from a trade-out — Indian STT on exercised
    /// options is the standard example.
    pub settlement_fee: f64,
    /// How the contract settles at its expiry: cash or physical delivery, and
    /// where the settlement price comes from. The default settles in cash at
    /// intrinsic value when the underlying's price is supplied, else at the
    /// last price, and says which on the settlement event.
    #[serde(default)]
    pub settlement: SettlementTerms,
    /// Risk-scenario (SPAN-style) margin for a SHORT option, as a fraction of
    /// the underlying notional at the strike. `0.0` (default) leaves a short
    /// option funded at its premium, as before.
    ///
    /// A sold option can lose far more than the premium it collects, so an
    /// exchange blocks a deposit scaled to the underlying's value, not to the
    /// premium. The kernel has no spot series, so the strike stands in for
    /// spot — the at-the-money case, which is the largest requirement and
    /// therefore the safe side to err on.
    #[serde(default)]
    pub span_pct: f64,
    /// Exposure margin for a SHORT option, as a fraction of the underlying
    /// notional at the strike, charged on top of `span_pct`. `0.0` = none.
    #[serde(default)]
    pub exposure_pct: f64,
    /// Listing venue (an exchange code such as `"NSE"`). Metadata only.
    #[serde(default)]
    pub venue: Option<String>,
    /// Quote currency (`"INR"`). Metadata only: the engine does no currency
    /// conversion, and every amount stays in the units of the price series.
    #[serde(default)]
    pub currency: Option<String>,
    /// Trading calendar. `None` (default) leaves the instrument tradable at
    /// every timestamp, exactly as before; with a schedule the kernel refuses
    /// orders outside the session, skips matching outside it, squares off
    /// at the schedule's square-off minute and expires DAY orders at the close.
    #[serde(default)]
    pub session: Option<SessionSchedule>,
    /// The exchange's classification of the scrip where charges depend on it
    /// (BSE cash equity: `"A"`, `"B"`, ...). `None` where it does not matter.
    #[serde(default)]
    pub scrip_group: Option<String>,
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
            settlement_fee: 0.0,
            settlement: SettlementTerms::default(),
            span_pct: 0.0,
            exposure_pct: 0.0,
            venue: None,
            currency: None,
            session: None,
            scrip_group: None,
        }
    }

    /// Initial margin per contract for a SHORT option under the SPAN-style
    /// model, or `None` when the instrument is not an option or the model is
    /// off (`span_pct` and `exposure_pct` both zero).
    ///
    /// `(span_pct + exposure_pct) × strike × multiplier`: the premium received
    /// is not part of the requirement — it stays in the balance, the way a
    /// broker credits it and blocks the deposit separately.
    pub fn short_option_margin_per_contract(&self) -> Option<f64> {
        let rate = self.span_pct + self.exposure_pct;
        if rate <= 0.0 {
            return None;
        }
        match &self.kind {
            InstrumentKind::Option { strike, .. } => Some(rate * strike * self.multiplier),
            _ => None,
        }
    }

    /// Whether the exchange segment the instrument trades on holds a good-till-
    /// cancelled or good-till-date order itself: `Some(true)` on NSE equity
    /// cash and NSE commodity (a date no later than the contract's maturity),
    /// `Some(false)` where only day orders are established (NSE equity
    /// derivatives, currency, BSE, MCX), `None` without a venue or for one the engine
    /// does not know, where it allows the order. A broker that
    /// keeps an order alive by resubmitting it each day is the host's business,
    /// not the exchange's validity.
    pub fn native_gtc(&self) -> Option<bool> {
        match self.venue.as_deref()?.to_ascii_uppercase().as_str() {
            "NSE" | "NCO" => Some(true),
            "NFO" | "BFO" | "CDS" | "BCD" | "BSE" | "MCX" => Some(false),
            // A venue the engine knows nothing about: no claim either way.
            _ => None,
        }
    }

    /// The underlying's symbol, for a contract or option that names one.
    pub fn underlying(&self) -> Option<&str> {
        match &self.kind {
            InstrumentKind::Contract { underlying } | InstrumentKind::Option { underlying, .. } => {
                underlying.as_deref()
            }
            _ => None,
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

    /// Snap an executed fill price to the price grid.
    ///
    /// A market-style fill (`limit_priced == false`) snaps against the
    /// trader: a buy rounds up, a sell rounds down, so slippage can never
    /// produce a better-than-grid price. A limit or target fill
    /// (`limit_priced == true`) is expected to sit on the grid already; if it
    /// does not, it snaps in the trader's favour (buy down, sell up) so the
    /// fill never crosses its own limit.
    ///
    /// A price within float noise of a tick (`< 1e-9` ticks, relative) is
    /// treated as on the grid and returned untouched, so an on-grid price never
    /// moves by a whole tick. A zero increment is a no-op.
    pub fn snap_fill(&self, price: Price, buy: bool, limit_priced: bool) -> Price {
        if self.price_increment <= 0.0 {
            return price;
        }
        let ticks = price / self.price_increment;
        let nearest = ticks.round();
        if (ticks - nearest).abs() < 1e-9 * nearest.abs().max(1.0) {
            return price;
        }
        let snapped = if buy != limit_priced { ticks.ceil() } else { ticks.floor() };
        snapped * self.price_increment
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
    pub fn settlement_value(
        &self,
        fallback_price: Price,
        underlying_price: Option<Price>,
    ) -> Price {
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
        assert!(spec.venue.is_none() && spec.currency.is_none() && spec.session.is_none());
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
    fn fill_snaps_against_the_trader_and_limits_in_their_favour() {
        let spec = InstrumentSpec {
            price_increment: 0.05,
            ..InstrumentSpec::new("X", InstrumentKind::Cash)
        };
        let eq = |a: f64, b: f64| (a - b).abs() < 1e-9;
        // Market-style: buy up, sell down.
        assert!(eq(spec.snap_fill(100.01, true, false), 100.05));
        assert!(eq(spec.snap_fill(100.04, false, false), 100.0));
        // Limit-priced: buy down, sell up (never crosses the limit).
        assert!(eq(spec.snap_fill(100.04, true, true), 100.0));
        assert!(eq(spec.snap_fill(100.01, false, true), 100.05));
        // On-grid prices never move, including float-noise cases.
        for &p in &[100.05, 100.1, 99.95, 0.15, 1234.55] {
            for &(buy, limit) in &[(true, false), (false, false), (true, true), (false, true)] {
                assert!(eq(spec.snap_fill(p, buy, limit), p), "{p} {buy} {limit}");
            }
        }
        // No increment: bit-identical.
        let none = InstrumentSpec::new("X", InstrumentKind::Cash);
        assert_eq!(none.snap_fill(100.0123, true, false), 100.0123);
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
