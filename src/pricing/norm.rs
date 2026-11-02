//! The standard normal CDF and density with fixed constants and no platform `erfc`.
//!
//! `norm_cdf` is `0.5 * erfc(-x / sqrt(2))`, identical on every platform.
//! Rust's stable `std` has no `erfc`, so this file carries one: the FreeBSD msun `s_erf.c`
//! rational approximations (Sun Microsystems, 1993; the same coefficients as Go's
//! math/erf.go), written as their exact IEEE bit patterns. Its error is under one ulp.
//!
//! Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved. Developed at
//! SunPro, a Sun Microsystems, Inc. business. Permission to use, copy, modify, and
//! distribute this software is freely granted, provided that this notice is preserved.

const fn c(bits: u64) -> f64 {
    f64::from_bits(bits)
}

const ERX: f64 = c(0x3FEB0AC160000000);
// erf on [0, 0.84375]
const PP0: f64 = c(0x3FC06EBA8214DB68);
const PP1: f64 = c(0xBFD4CD7D691CB913);
const PP2: f64 = c(0xBF9D2A51DBD7194F);
const PP3: f64 = c(0xBF77A291236668E4);
const PP4: f64 = c(0xBEF8EAD6120016AC);
const QQ1: f64 = c(0x3FD97779CDDADC09);
const QQ2: f64 = c(0x3FB0A54C5536CEBA);
const QQ3: f64 = c(0x3F74D022C4D36B0F);
const QQ4: f64 = c(0x3F215DC9221C1A10);
const QQ5: f64 = c(0xBED09C4342A26120);
// erf on [0.84375, 1.25]
const PA0: f64 = c(0xBF6359B8BEF77538);
const PA1: f64 = c(0x3FDA8D00AD92B34D);
const PA2: f64 = c(0xBFD7D240FBB8C3F1);
const PA3: f64 = c(0x3FD45FCA805120E4);
const PA4: f64 = c(0xBFBC63983D3E28EC);
const PA5: f64 = c(0x3FA22A36599795EB);
const PA6: f64 = c(0xBF61BF380A96073F);
const QA1: f64 = c(0x3FBB3E6618EEE323);
const QA2: f64 = c(0x3FE14AF092EB6F33);
const QA3: f64 = c(0x3FB2635CD99FE9A7);
const QA4: f64 = c(0x3FC02660E763351F);
const QA5: f64 = c(0x3F8BEDC26B51DD1C);
const QA6: f64 = c(0x3F888B545735151D);
// erfc on [1.25, 1/0.35]
const RA0: f64 = c(0xBF843412600D6435);
const RA1: f64 = c(0xBFE63416E4BA7360);
const RA2: f64 = c(0xC0251E0441B0E726);
const RA3: f64 = c(0xC04F300AE4CBA38D);
const RA4: f64 = c(0xC0644CB184282266);
const RA5: f64 = c(0xC067135CEBCCABB2);
const RA6: f64 = c(0xC054526557E4D2F2);
const RA7: f64 = c(0xC023A0EFC69AC25C);
const SA1: f64 = c(0x4033A6B9BD707687);
const SA2: f64 = c(0x4061350C526AE721);
const SA3: f64 = c(0x407B290DD58A1A71);
const SA4: f64 = c(0x40842B1921EC2868);
const SA5: f64 = c(0x407AD02157700314);
const SA6: f64 = c(0x405B28A3EE48AE2C);
const SA7: f64 = c(0x401A47EF8E484A93);
const SA8: f64 = c(0xBFAEEFF2EE749A62);
// erfc on [1/0.35, 28]
const RB0: f64 = c(0xBF84341239E86F4A);
const RB1: f64 = c(0xBFE993BA70C285DE);
const RB2: f64 = c(0xC031C209555F995A);
const RB3: f64 = c(0xC064145D43C5ED98);
const RB4: f64 = c(0xC083EC881375F228);
const RB5: f64 = c(0xC09004616A2E5992);
const RB6: f64 = c(0xC07E384E9BDC383F);
const SB1: f64 = c(0x403E568B261D5190);
const SB2: f64 = c(0x40745CAE221B9F0A);
const SB3: f64 = c(0x409802EB189D5118);
const SB4: f64 = c(0x40A8FFB7688C246A);
const SB5: f64 = c(0x40A3F219CEDF3BE6);
const SB6: f64 = c(0x407DA874E79FE763);
const SB7: f64 = c(0xC03670E242712D62);

/// The complementary error function.
pub fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return 0.0;
    }
    if x == f64::NEG_INFINITY {
        return 2.0;
    }
    let sign = x < 0.0;
    let x = x.abs();
    if x < 0.84375 {
        let temp = if x < 1.0 / (1u64 << 56) as f64 {
            x
        } else {
            let z = x * x;
            let r = PP0 + z * (PP1 + z * (PP2 + z * (PP3 + z * PP4)));
            let s = 1.0 + z * (QQ1 + z * (QQ2 + z * (QQ3 + z * (QQ4 + z * QQ5))));
            let y = r / s;
            if x < 0.25 {
                x + x * y
            } else {
                0.5 + (x * y + (x - 0.5))
            }
        };
        return if sign { 1.0 + temp } else { 1.0 - temp };
    }
    if x < 1.25 {
        let s = x - 1.0;
        let p = PA0 + s * (PA1 + s * (PA2 + s * (PA3 + s * (PA4 + s * (PA5 + s * PA6)))));
        let q = 1.0 + s * (QA1 + s * (QA2 + s * (QA3 + s * (QA4 + s * (QA5 + s * QA6)))));
        return if sign { 1.0 + ERX + p / q } else { 1.0 - ERX - p / q };
    }
    if x < 28.0 {
        let s = 1.0 / (x * x);
        let (r, ss) = if x < 1.0 / 0.35 {
            (
                RA0 + s
                    * (RA1 + s * (RA2 + s * (RA3 + s * (RA4 + s * (RA5 + s * (RA6 + s * RA7)))))),
                1.0 + s
                    * (SA1
                        + s * (SA2
                            + s * (SA3 + s * (SA4 + s * (SA5 + s * (SA6 + s * (SA7 + s * SA8))))))),
            )
        } else {
            if sign && x > 6.0 {
                return 2.0;
            }
            (
                RB0 + s * (RB1 + s * (RB2 + s * (RB3 + s * (RB4 + s * (RB5 + s * RB6))))),
                1.0 + s
                    * (SB1 + s * (SB2 + s * (SB3 + s * (SB4 + s * (SB5 + s * (SB6 + s * SB7)))))),
            )
        };
        // x with its low 32 bits cleared, so exp(-z*z) is exact enough
        let z = f64::from_bits(x.to_bits() & 0xffff_ffff_0000_0000);
        let e = (-z * z - 0.5625).exp() * ((z - x) * (z + x) + r / ss).exp();
        return if sign { 2.0 - e / x } else { e / x };
    }
    if sign {
        2.0
    } else {
        0.0
    }
}

/// Standard normal CDF, `0.5 * erfc(-x / sqrt(2))`.
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

/// Standard normal density, `exp(-x^2/2) / sqrt(2 pi)`.
pub fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Standard normal quantile by 200 bisection steps on [`norm_cdf`] over [-40, 40]:
/// slow, exact to 1e-12, and the same on every platform. `None` outside (0, 1).
pub fn norm_ppf(p: f64) -> Option<f64> {
    if !(p > 0.0 && p < 1.0) {
        return None;
    }
    let (mut lo, mut hi) = (-40.0f64, 40.0f64);
    for _ in 0..200 {
        let mid = (lo + hi) / 2.0;
        if norm_cdf(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some((lo + hi) / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference values from an arbitrary-precision evaluation of
    // 0.5 * erfc(-x / sqrt(2)).
    const CDF: &[(f64, f64)] = &[
        (0.0, 0.5),
        (1.96, 0.9750021048517795),
        (-1.96, 0.02499789514822043),
        (1.0, 0.8413447460685429),
        (-3.0, 0.0013498980316300957),
        (5.0, 0.9999997133484281),
        (-8.5, 9.479534822203356e-18),
        (0.3, 0.6179114221889526),
        (2.5, 0.9937903346742238),
        (10.0, 1.0),
    ];

    #[test]
    fn cdf_matches_reference() {
        for &(x, want) in CDF {
            let got = norm_cdf(x);
            assert!(
                (got - want).abs() <= 1e-15 * want.abs().max(1e-300),
                "norm_cdf({x}) = {got:e}, reference {want:e}"
            );
        }
    }

    #[test]
    fn cdf_is_symmetric_and_pdf_integrates_to_one() {
        for i in -60..=60 {
            let x = i as f64 * 0.1;
            assert!((norm_cdf(x) + norm_cdf(-x) - 1.0).abs() < 1e-15, "{x}");
        }
        let h = 1e-3;
        let total: f64 = (-8000..8000).map(|i| norm_pdf((i as f64 + 0.5) * h) * h).sum();
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn nan_stays_nan() {
        assert!(norm_cdf(f64::NAN).is_nan());
    }

    #[test]
    fn ppf_matches_reference() {
        let cases = [
            (0.975, 1.9599639845400532),
            (0.025, -1.959963984540054),
            (1e-10, -6.3613409024040575),
            (0.999, 3.090232306167797),
            (0.123, -1.1601198829975194),
        ];
        for (p, want) in cases {
            let got = norm_ppf(p).unwrap();
            assert!((got - want).abs() < 1e-12, "norm_ppf({p}) = {got}, reference {want}");
        }
        assert!(norm_ppf(0.0).is_none() && norm_ppf(1.0).is_none() && norm_ppf(f64::NAN).is_none());
    }
}
