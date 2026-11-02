//! Where money is rounded.
//!
//! A computed amount is `f64`, so a value that is exactly half a paisa in decimal
//! (0.025% of 1,500.00 is 0.375) is rarely exactly half a paisa in binary.
//! Rounding it with `f64::round` would tip either way on representation
//! noise. [`round_places`] first treats a value within a few ulps of a tie as
//! the tie it was meant to be, then applies the named rule.
//!
//! This is the one place an amount is rounded: charges to the paisa, and every
//! `f64` amount to the integer minor units it is booked in ([`to_minor`]).
//! The rule for each boundary is in `docs/quantitative-correctness/numerics.md`.

/// How a tie (an exact half) goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Round {
    /// Away from zero.
    HalfUp,
    /// To the even neighbour.
    HalfEven,
}

/// Relative distance from a tie within which a value counts as that tie.
/// A product of three or four rates and amounts carries at most a few ulps of
/// error; sixteen is wide enough for that and far narrower than any real
/// fraction of a paisa.
const TIE_ULPS: f64 = 16.0;

/// A non-negative scaled value rounded to a whole number by `mode`, with a
/// value within [`TIE_ULPS`] of a tie treated as that tie.
fn round_scaled(scaled: f64, mode: Round) -> f64 {
    let floor = scaled.floor();
    let frac = scaled - floor;
    let tie = (frac - 0.5).abs() <= TIE_ULPS * f64::EPSILON * scaled.max(1.0);
    if tie {
        match mode {
            Round::HalfUp => floor + 1.0,
            Round::HalfEven => {
                if floor % 2.0 == 0.0 {
                    floor
                } else {
                    floor + 1.0
                }
            }
        }
    } else {
        scaled.round()
    }
}

/// `x` rounded to `places` decimal places.
pub fn round_places(x: f64, places: u32, mode: Round) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let scale = 10f64.powi(places as i32);
    (round_scaled(x.abs() * scale, mode) / scale).copysign(x)
}

/// The largest magnitude, in minor units, a booked amount may have: 2^120,
/// about 1.3e36. Minor units are `i128`, so the sum of any two booked amounts
/// stays far inside the type and a run on garbage prices books garbage
/// amounts without wrapping; only a value past this bound, which no price and
/// size reach, is refused.
pub const MAX_MINOR: i128 = 1 << 120;

/// `10^n` for the scales a booked amount may use; exact in `f64`.
const POW10: [f64; 10] = [1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9];
const MAX_POW10: u32 = 9;

/// `x` in minor units of `places` decimals (a rupee at `places = 2` is 100),
/// rounded by `mode`: the one conversion from an `f64` amount to a booked
/// integer. `None` when `x` is not finite or its rounded value exceeds
/// [`MAX_MINOR`].
pub fn to_minor(x: f64, places: u32, mode: Round) -> Option<i128> {
    if !x.is_finite() {
        return None;
    }
    let scale = POW10[places.min(MAX_POW10) as usize];
    let rounded = round_scaled(x.abs() * scale, mode);
    if rounded > MAX_MINOR as f64 {
        return None;
    }
    // `f64 -> i64` is one instruction, `f64 -> i128` a library call.
    let minor = if rounded < 9.0e18 { rounded as i64 as i128 } else { rounded as i128 };
    Some(if x.is_sign_negative() { -minor } else { minor })
}

/// `x` to the paisa (two places).
#[inline]
pub fn paise(x: f64, mode: Round) -> f64 {
    round_places(x, 2, mode)
}

/// `x` to a whole rupee, half up (the contract note's stamp duty).
#[inline]
pub fn whole_rupees(x: f64) -> f64 {
    round_places(x, 0, Round::HalfUp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ties_go_by_the_named_rule_despite_binary_noise() {
        // 0.00025 * 1500 is 0.375 in decimal; in binary it may not be.
        let x = 0.00025 * 1500.0;
        assert_eq!(paise(x, Round::HalfUp), 0.38);
        assert_eq!(paise(x, Round::HalfEven), 0.38); // 37 is odd, so up to 38
        assert_eq!(paise(0.385, Round::HalfEven), 0.38); // 38 is even
        assert_eq!(paise(0.385, Round::HalfUp), 0.39);
        assert_eq!(paise(1.005, Round::HalfUp), 1.01);
        assert_eq!(paise(2.675, Round::HalfEven), 2.68);
    }

    #[test]
    fn non_ties_round_to_the_nearest_paisa() {
        assert_eq!(paise(1.234, Round::HalfUp), 1.23);
        assert_eq!(paise(1.236, Round::HalfEven), 1.24);
        assert_eq!(paise(0.0, Round::HalfUp), 0.0);
        assert_eq!(whole_rupees(2.5), 3.0);
        assert_eq!(whole_rupees(2.49), 2.0);
    }

    #[test]
    fn large_values_keep_their_tie() {
        assert_eq!(paise(123_456_789.125, Round::HalfUp), 123_456_789.13);
        assert_eq!(paise(123_456_789.125, Round::HalfEven), 123_456_789.12);
    }

    #[test]
    fn to_minor_rounds_by_rule_and_refuses_the_unbookable() {
        assert_eq!(to_minor(1.005, 2, Round::HalfEven), Some(100)); // 100 is even
        assert_eq!(to_minor(1.015, 2, Round::HalfEven), Some(102));
        assert_eq!(to_minor(1.005, 2, Round::HalfUp), Some(101));
        assert_eq!(to_minor(-2.675, 2, Round::HalfEven), Some(-268));
        assert_eq!(to_minor(12.5, 0, Round::HalfEven), Some(12));
        assert_eq!(to_minor(0.0, 2, Round::HalfEven), Some(0));
        assert_eq!(to_minor(f64::NAN, 2, Round::HalfEven), None);
        assert_eq!(to_minor(f64::INFINITY, 2, Round::HalfEven), None);
        assert_eq!(to_minor(1e36, 2, Round::HalfEven), None);
        assert_eq!(
            to_minor(1e30, 0, Round::HalfEven),
            Some(1_000_000_000_000_000_019_884_624_838_656)
        );
    }

    #[test]
    fn sign_and_non_finite_pass_through() {
        assert_eq!(paise(-0.385, Round::HalfUp), -0.39);
        assert!(paise(f64::NAN, Round::HalfUp).is_nan());
        assert_eq!(paise(f64::INFINITY, Round::HalfUp), f64::INFINITY);
    }
}
