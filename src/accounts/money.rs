//! Booked money: scaled integers at the accounting boundary.
//!
//! Cash, fees, the notional of a fill and realised P&L are held as [`Money`],
//! an `i128` count of minor units, at a [`MoneyScale`] declared once per run
//! (two decimals by default: a rupee or a dollar is 100 minor units). Sums of
//! booked amounts are then exact integer sums, so cash, the cost basis of the
//! open positions and realised P&L reconcile to the unit, and a million fills
//! add no drift.
//!
//! What stays `f64`: prices, quantities, marks, margin requirements,
//! statistics, indicators and the equity curve. They are not booked amounts.
//! An amount enters the books once, through [`MoneyScale::money`], which
//! rounds by [`Round`] (half to even unless the caller names another rule) and
//! refuses what cannot be booked (not finite, or beyond
//! [`crate::execution::costs::rounding::MAX_MINOR`]) instead of clamping it.
//! It leaves as an `f64` view through [`MoneyScale::to_f64`], which is
//! `minor / 10^places` and so the double nearest the exact decimal.
//!
//! Rules per boundary are in `docs/quantitative-correctness/numerics.md`.

use std::ops::{Add, AddAssign, Neg, Sub, SubAssign};

use crate::core::error::{RaptorError, Result};
use crate::execution::costs::rounding::{to_minor, Round};

/// The most decimal places a scale may declare: nine is a nano-unit, finer than
/// any price tick, and keeps `10^places` exact as an `f64`.
pub const MAX_PLACES: u8 = 9;

/// The one way a booked amount fails mid-run: an amount that cannot be booked
/// (not finite, or past the minor-unit range) or a sum that overflows. It
/// panics with a message that starts `money_overflow:`, which the Python layer
/// (`raptorbt._guard`) turns into the structured error of that code at every
/// entry point. Rust callers of the kernel see the panic.
#[cold]
#[inline(never)]
pub fn money_fault(detail: impl std::fmt::Display) -> ! {
    panic!("{MONEY_FAULT_PREFIX} {detail}")
}

/// What a [`money_fault`] message starts with.
pub const MONEY_FAULT_PREFIX: &str = "money_overflow:";

/// An amount of money in minor units of a [`MoneyScale`].
///
/// Arithmetic panics on overflow rather than wrapping: a wrapped cash balance
/// would be silently wrong. An `i128` of minor units holds about 1.7e36 whole
/// units at two decimals, past anything a price and a size can book.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Money(i128);

impl Money {
    pub const ZERO: Money = Money(0);

    #[inline]
    pub const fn from_minor(minor: i128) -> Self {
        Money(minor)
    }

    #[inline]
    pub const fn minor(self) -> i128 {
        self.0
    }

    #[inline]
    pub fn is_negative(self) -> bool {
        self.0 < 0
    }

    #[inline]
    pub fn abs(self) -> Money {
        Money(
            self.0
                .checked_abs()
                .unwrap_or_else(|| money_fault("|amount| exceeds i128 minor units")),
        )
    }

    /// The part of this amount that `fraction` of the whole stands for,
    /// rounded half to even. `fraction` must be in `[0, 1]`; the callers
    /// split a position by the quantity closed over the quantity held. The
    /// product is taken in `f64`, so it is exact while the amount is under
    /// 2^53 minor units (9e13 whole units at two decimals). The remainder is
    /// `self - part`, so the two parts add back to the whole exactly.
    pub fn portion(self, fraction: f64) -> Money {
        debug_assert!((0.0..=1.0).contains(&fraction), "portion fraction {fraction}");
        let part = to_minor(self.0 as f64 * fraction, 0, Round::HalfEven)
            .unwrap_or_else(|| money_fault("a share of a booked amount cannot be booked"));
        Money(part)
    }
}

impl Add for Money {
    type Output = Money;
    #[inline]
    fn add(self, rhs: Money) -> Money {
        Money(
            self.0
                .checked_add(rhs.0)
                .unwrap_or_else(|| money_fault("sum exceeds i128 minor units")),
        )
    }
}

impl Sub for Money {
    type Output = Money;
    #[inline]
    fn sub(self, rhs: Money) -> Money {
        Money(
            self.0
                .checked_sub(rhs.0)
                .unwrap_or_else(|| money_fault("difference exceeds i128 minor units")),
        )
    }
}

impl Neg for Money {
    type Output = Money;
    #[inline]
    fn neg(self) -> Money {
        Money(
            self.0
                .checked_neg()
                .unwrap_or_else(|| money_fault("negation exceeds i128 minor units")),
        )
    }
}

impl AddAssign for Money {
    #[inline]
    fn add_assign(&mut self, rhs: Money) {
        *self = *self + rhs;
    }
}

impl SubAssign for Money {
    #[inline]
    fn sub_assign(&mut self, rhs: Money) {
        *self = *self - rhs;
    }
}

impl std::iter::Sum for Money {
    fn sum<I: Iterator<Item = Money>>(iter: I) -> Money {
        iter.fold(Money::ZERO, |a, b| a + b)
    }
}

/// The decimals a run books money in (the same for cash, fees and P&L).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MoneyScale {
    places: u8,
}

/// `10^n` for the view back to `f64`, exact for every allowed scale.
const POW10: [f64; MAX_PLACES as usize + 1] = [1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9];

impl Default for MoneyScale {
    /// Two decimals: INR, USD and most currencies.
    fn default() -> Self {
        MoneyScale { places: 2 }
    }
}

impl MoneyScale {
    pub fn new(places: u32) -> Result<Self> {
        if places > u32::from(MAX_PLACES) {
            return Err(RaptorError::invalid_parameter(format!(
                "money_decimals must be at most {MAX_PLACES}, got {places}"
            )));
        }
        Ok(MoneyScale { places: places as u8 })
    }

    #[inline]
    pub fn places(self) -> u32 {
        u32::from(self.places)
    }

    /// The amount `x` booked at this scale, rounded by `mode`.
    pub fn money_by(self, x: f64, mode: Round) -> Result<Money> {
        to_minor(x, self.places(), mode).map(Money).ok_or_else(|| {
            RaptorError::invalid_parameter(format!(
                "amount {x} cannot be booked at {} decimals (not finite, or beyond the \
                 minor-unit range)",
                self.places
            ))
        })
    }

    /// The amount `x` booked at this scale, half to even.
    #[inline]
    pub fn money(self, x: f64) -> Result<Money> {
        self.money_by(x, Round::HalfEven)
    }

    /// The `f64` view of an amount: the double nearest `minor / 10^places`.
    #[inline]
    pub fn to_f64(self, m: Money) -> f64 {
        // A balance fits an `i64` outside absurd runs, and `i64 -> f64` is one
        // instruction where `i128 -> f64` is a library call. The wide arm is out
        // of line: inlined, the compiler computes both arms and selects, which
        // makes the library call on every view. Both arms round the same integer
        // to nearest, so the value is the same either way.
        let minor = match i64::try_from(m.0) {
            Ok(small) => small as f64,
            Err(_) => wide_to_f64(m.0),
        };
        minor / POW10[self.places as usize]
    }

    /// `x` as the `f64` view of its booked amount (rounded to the scale).
    /// For a value the books will hold anyway, so that the number a trade
    /// reports is the number the cash moved by.
    #[inline]
    pub fn snap(self, x: f64) -> Result<f64> {
        Ok(self.to_f64(self.money(x)?))
    }
}

/// `i128 -> f64` for an amount beyond `i64`, kept off the path of every
/// other conversion (see [`MoneyScale::to_f64`]).
#[cold]
#[inline(never)]
fn wide_to_f64(minor: i128) -> f64 {
    minor as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn booking_rounds_half_even_and_views_exactly() {
        let s = MoneyScale::default();
        assert_eq!(s.money(0.125).unwrap().minor(), 12);
        assert_eq!(s.money(0.135).unwrap().minor(), 14);
        assert_eq!(s.money(-1234.567).unwrap().minor(), -123_457);
        assert_eq!(s.to_f64(Money::from_minor(-123_457)), -1234.57);
        assert_eq!(s.money_by(0.125, Round::HalfUp).unwrap().minor(), 13);
    }

    #[test]
    fn unbookable_amounts_are_errors() {
        let s = MoneyScale::default();
        assert!(s.money(f64::NAN).is_err());
        assert!(s.money(f64::NEG_INFINITY).is_err());
        assert!(s.money(1e37).is_err());
        assert!(s.money(1e30).is_ok());
        assert!(MoneyScale::new(10).is_err());
        assert_eq!(MoneyScale::new(0).unwrap().money(2.5).unwrap().minor(), 2);
    }

    #[test]
    fn a_view_round_trips_through_booking() {
        let s = MoneyScale::default();
        for minor in [-9_007_199_254_740_991_i128 / 3, -5, -1, 0, 1, 7, 99_999_999_999] {
            let m = Money::from_minor(minor);
            assert_eq!(s.money(s.to_f64(m)).unwrap(), m, "{minor}");
        }
    }

    #[test]
    fn a_view_is_the_wide_conversion_on_both_sides_of_i64() {
        let edge = i128::from(i64::MAX);
        let low = i128::from(i64::MIN);
        for minor in [0, -1, 1, 9_007_199_254_740_993, edge, edge + 1, low, low - 1, edge * 1_000] {
            for places in [0, 2, 9] {
                let s = MoneyScale::new(places).unwrap();
                let direct = minor as f64 / POW10[places as usize];
                assert_eq!(
                    s.to_f64(Money::from_minor(minor)).to_bits(),
                    direct.to_bits(),
                    "{minor}"
                );
            }
        }
    }

    #[test]
    fn portions_add_back_to_the_whole() {
        for (total, f) in [(10_001_i128, 1.0 / 3.0), (-777, 0.5), (1, 0.5), (3, 0.5), (0, 0.9)] {
            let whole = Money::from_minor(total);
            let part = whole.portion(f);
            assert_eq!(part + (whole - part), whole);
        }
        assert_eq!(Money::from_minor(5).portion(0.5).minor(), 2); // half to even
        assert_eq!(Money::from_minor(3).portion(0.5).minor(), 2);
    }

    #[test]
    #[should_panic(expected = "money_overflow:")]
    fn overflow_is_loud() {
        let _ = Money::from_minor(i128::MAX) + Money::from_minor(1);
    }
}
