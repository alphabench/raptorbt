//! Where money is rounded.
//!
//! Money is `f64` for now, so a value that is exactly half a paisa in decimal
//! (0.025% of 1,500.00 is 0.375) is rarely exactly half a paisa in binary.
//! Rounding it with `f64::round` would tip either way on representation
//! noise. [`round_places`] first treats a value within a few ulps of a tie as
//! the tie it was meant to be, then applies the named rule.
//!
//! This is the one place a charge is rounded. When accounting moves to fixed
//! point, only this module changes. The rule for each boundary is in
//! `docs/quantitative-correctness/numerics.md`.

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

/// `x` rounded to `places` decimal places.
pub fn round_places(x: f64, places: u32, mode: Round) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let scale = 10f64.powi(places as i32);
    let scaled = x.abs() * scale;
    let floor = scaled.floor();
    let frac = scaled - floor;
    let tie = (frac - 0.5).abs() <= TIE_ULPS * f64::EPSILON * scaled.max(1.0);
    let rounded = if tie {
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
    };
    (rounded / scale).copysign(x)
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
    fn sign_and_non_finite_pass_through() {
        assert_eq!(paise(-0.385, Round::HalfUp), -0.39);
        assert!(paise(f64::NAN, Round::HalfUp).is_nan());
        assert_eq!(paise(f64::INFINITY, Round::HalfUp), f64::INFINITY);
    }
}
