//! Special functions against the committed scipy fixture (relative 1e-12) and against closed
//! forms that need no reference library.

use raptorbt::math::special::*;
use raptorbt::pricing::norm::erfc;
use serde_json::Value;

const FIXTURE: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/tests/python/oracle/fixtures/stats/stats_special.json");
const REL: f64 = 1e-12;

fn fixture() -> Value {
    let text = std::fs::read_to_string(FIXTURE).expect("fixture stats_special.json");
    serde_json::from_str(&text).expect("fixture parses")
}

/// A float stored as the 16 hex digits of its big-endian IEEE bits.
fn hex(v: &Value) -> f64 {
    f64::from_bits(u64::from_str_radix(v.as_str().expect("hex string"), 16).expect("hex digits"))
}

fn rel(got: f64, want: f64) -> f64 {
    if want == 0.0 {
        got.abs()
    } else {
        ((got - want) / want).abs()
    }
}

fn close(what: &str, got: f64, want: f64, tol: f64) {
    assert!(rel(got, want) <= tol, "{what}: got {got:e}, want {want:e}, rel {:e}", rel(got, want));
}

fn cases<'a>(fx: &'a Value, family: &str) -> &'a Vec<Value> {
    let rows = fx["cases"][family].as_array().expect("family rows");
    assert!(!rows.is_empty(), "{family}: no fixture rows");
    rows
}

#[test]
fn ln_gamma_matches_the_fixture() {
    let fx = fixture();
    for c in cases(&fx, "ln_gamma") {
        let x = hex(&c["x"]);
        close(&format!("ln_gamma({x})"), ln_gamma(x).unwrap(), hex(&c["y"]), REL);
    }
}

#[test]
fn ln_gamma_at_one_and_two_is_exactly_zero() {
    assert_eq!(ln_gamma(1.0).unwrap(), 0.0);
    assert_eq!(ln_gamma(2.0).unwrap(), 0.0);
}

#[test]
fn ln_gamma_closed_forms() {
    // G(1/2) = sqrt(pi)
    close("ln G(1/2)", ln_gamma(0.5).unwrap(), 0.5 * std::f64::consts::PI.ln(), REL);
    // G(n) = (n - 1)!
    let mut factorial = 1.0f64;
    for n in 3..=20u32 {
        factorial *= (n - 1) as f64;
        close(&format!("ln G({n})"), ln_gamma(n as f64).unwrap(), factorial.ln(), REL);
    }
    // the recurrence G(x + 1) = x G(x), across the point where the series takes over
    for x in [0.3, 1.7, 7.5, 14.5, 15.5, 40.25] {
        let lhs = ln_gamma(x + 1.0).unwrap();
        let rhs = x.ln() + ln_gamma(x).unwrap();
        assert!((lhs - rhs).abs() <= 1e-13 * lhs.abs().max(1.0), "recurrence at {x}");
    }
}

#[test]
fn incomplete_beta_matches_the_fixture() {
    let fx = fixture();
    for c in cases(&fx, "inc_beta") {
        let (a, b, x) = (hex(&c["a"]), hex(&c["b"]), hex(&c["x"]));
        let (p, q) = inc_beta_pq(a, b, x, 1.0 - x).unwrap();
        close(&format!("I_{x}({a}, {b})"), p, hex(&c["p"]), REL);
        close(&format!("1 - I_{x}({a}, {b})"), q, hex(&c["q"]), REL);
        assert!((p + q - 1.0).abs() <= 4e-16, "P + Q = 1 at ({a}, {b}, {x})");
    }
}

#[test]
fn incomplete_beta_closed_forms() {
    for x in [0.0, 1e-9, 0.2, 0.5, 0.97, 1.0] {
        close("I_x(1, 1) = x", inc_beta(1.0, 1.0, x).unwrap(), x, REL);
        // I_x(a, 1) = x^a and I_x(1, b) = 1 - (1 - x)^b
        close("I_x(3, 1)", inc_beta(3.0, 1.0, x).unwrap(), x.powi(3), REL);
        if x < 1.0 {
            let want = -(4.0 * (-x).ln_1p()).exp_m1();
            close("I_x(1, 4)", inc_beta(1.0, 4.0, x).unwrap(), want, REL);
        }
        // symmetry I_x(a, b) = 1 - I_{1-x}(b, a)
        let lhs = inc_beta(2.5, 7.0, x).unwrap();
        let rhs = 1.0 - inc_beta(7.0, 2.5, 1.0 - x).unwrap();
        assert!((lhs - rhs).abs() <= 1e-14, "symmetry at {x}");
    }
}

#[test]
fn incomplete_gamma_matches_the_fixture() {
    let fx = fixture();
    for c in cases(&fx, "inc_gamma") {
        let (a, x) = (hex(&c["a"]), hex(&c["x"]));
        let (p, q) = inc_gamma_pq(a, x).unwrap();
        close(&format!("P({a}, {x})"), p, hex(&c["p"]), REL);
        close(&format!("Q({a}, {x})"), q, hex(&c["q"]), REL);
        assert!((p + q - 1.0).abs() <= 4e-16, "P + Q = 1 at ({a}, {x})");
    }
}

#[test]
fn incomplete_gamma_closed_forms() {
    for x in [1e-6, 0.3, 1.0, 4.0, 20.0, 60.0] {
        // P(1, x) = 1 - e^-x and P(2, x) = 1 - e^-x (1 + x)
        close("P(1, x)", inc_gamma_p(1.0, x).unwrap(), -(-x).exp_m1(), REL);
        close("Q(2, x)", inc_gamma_q(2.0, x).unwrap(), (-x).exp() * (1.0 + x), REL);
        // Q(1/2, x) = erfc(sqrt(x))
        close("Q(1/2, x)", inc_gamma_q(0.5, x).unwrap(), erfc(x.sqrt()), 1e-12);
    }
    assert_eq!(inc_gamma_pq(3.0, 0.0).unwrap(), (0.0, 1.0));
    assert_eq!(inc_gamma_pq(3.0, f64::INFINITY).unwrap(), (1.0, 0.0));
}

#[test]
fn t_distribution_matches_the_fixture() {
    let fx = fixture();
    for c in cases(&fx, "t") {
        let (x, df) = (hex(&c["x"]), hex(&c["df"]));
        close(&format!("t_cdf({x}, {df})"), t_cdf(x, df).unwrap(), hex(&c["cdf"]), REL);
        close(&format!("t_sf({x}, {df})"), t_sf(x, df).unwrap(), hex(&c["sf"]), REL);
    }
}

#[test]
fn t_distribution_closed_forms() {
    for x in [-30.0, -2.5, -0.1, 0.0, 0.7, 3.0, 50.0] {
        // one degree of freedom is the Cauchy distribution
        close(
            "t_cdf df=1",
            t_cdf(x, 1.0).unwrap(),
            0.5 + x.atan() / std::f64::consts::PI,
            if x < -2.0 { 1e-11 } else { REL },
        );
        // two degrees of freedom: 1/2 + x / (2 sqrt(2 + x^2))
        let want = 0.5 + x / (2.0 * (2.0 + x * x).sqrt());
        if want > 1e-3 {
            close("t_cdf df=2", t_cdf(x, 2.0).unwrap(), want, REL);
        }
        let two = t_two_sided_p(x, 7.0).unwrap();
        close("two-sided = 2 sf(|x|)", two, 2.0 * t_sf(x.abs(), 7.0).unwrap(), 1e-14);
    }
    assert_eq!(t_cdf(0.0, 9.0).unwrap(), 0.5);
    assert_eq!(t_sf(f64::INFINITY, 9.0).unwrap(), 0.0);
    assert_eq!(t_sf(f64::NEG_INFINITY, 9.0).unwrap(), 1.0);
    // an x whose square overflows is a tail, not an error and not NaN
    let tail = t_sf(1e200, 5.0).unwrap();
    assert!((0.0..=1e-300).contains(&tail), "t_sf(1e200, 5) = {tail:e}");
    assert_eq!(t_cdf(-1e200, 5.0).unwrap(), tail);
}

#[test]
fn chi_squared_matches_the_fixture() {
    let fx = fixture();
    for c in cases(&fx, "chi2") {
        let (x, df) = (hex(&c["x"]), hex(&c["df"]));
        close(&format!("chi2_cdf({x}, {df})"), chi2_cdf(x, df).unwrap(), hex(&c["cdf"]), REL);
        close(&format!("chi2_sf({x}, {df})"), chi2_sf(x, df).unwrap(), hex(&c["sf"]), REL);
    }
}

#[test]
fn chi_squared_closed_forms() {
    for x in [0.01, 0.5, 2.0, 9.0, 40.0] {
        // two degrees of freedom is the exponential distribution with mean 2
        close("chi2_sf df=2", chi2_sf(x, 2.0).unwrap(), (-x / 2.0).exp(), REL);
        close("chi2_cdf df=2", chi2_cdf(x, 2.0).unwrap(), -(-x / 2.0).exp_m1(), REL);
        // one degree of freedom: erfc(sqrt(x / 2))
        close("chi2_sf df=1", chi2_sf(x, 1.0).unwrap(), erfc((x / 2.0).sqrt()), 1e-12);
    }
    assert_eq!(chi2_cdf(-1.0, 3.0).unwrap(), 0.0);
    assert_eq!(chi2_sf(0.0, 3.0).unwrap(), 1.0);
}

#[test]
fn f_distribution_matches_the_fixture() {
    let fx = fixture();
    for c in cases(&fx, "f") {
        let (x, d1, d2) = (hex(&c["x"]), hex(&c["d1"]), hex(&c["d2"]));
        close(&format!("f_cdf({x}, {d1}, {d2})"), f_cdf(x, d1, d2).unwrap(), hex(&c["cdf"]), REL);
        close(&format!("f_sf({x}, {d1}, {d2})"), f_sf(x, d1, d2).unwrap(), hex(&c["sf"]), REL);
    }
}

#[test]
fn f_distribution_closed_forms() {
    // two numerator degrees of freedom: sf(x) = (1 + 2 x / d2)^(-d2 / 2)
    for d2 in [3.0f64, 10.0, 77.0] {
        for x in [0.05f64, 0.5, 2.0, 11.0] {
            let want = (1.0 + 2.0 * x / d2).powf(-d2 / 2.0);
            close("f_sf d1=2", f_sf(x, 2.0, d2).unwrap(), want, 1e-12);
        }
    }
    // t^2 with df degrees of freedom is F(1, df)
    for x in [0.3, 1.5, 4.0] {
        let via_t = t_two_sided_p(x, 12.0).unwrap();
        close("F(1, 12) at t^2", f_sf(x * x, 1.0, 12.0).unwrap(), via_t, 1e-12);
    }
}

#[test]
fn out_of_domain_arguments_are_typed_errors() {
    assert!(matches!(ln_gamma(0.0), Err(SpecialError::Domain { .. })));
    assert!(matches!(ln_gamma(-3.5), Err(SpecialError::Domain { .. })));
    assert!(matches!(ln_gamma(f64::INFINITY), Err(SpecialError::Domain { .. })));
    assert!(matches!(ln_gamma(f64::NAN), Err(SpecialError::NotANumber { .. })));
    assert!(matches!(inc_beta(0.0, 1.0, 0.5), Err(SpecialError::Domain { .. })));
    assert!(matches!(inc_beta(1.0, 1.0, 1.5), Err(SpecialError::Domain { .. })));
    assert!(matches!(inc_beta(1.0, 1.0, f64::NAN), Err(SpecialError::NotANumber { .. })));
    assert!(matches!(inc_gamma_p(1.0, -1.0), Err(SpecialError::Domain { .. })));
    assert!(matches!(inc_gamma_p(-1.0, 1.0), Err(SpecialError::Domain { .. })));
    assert!(matches!(t_cdf(1.0, 0.0), Err(SpecialError::Domain { .. })));
    assert!(matches!(t_cdf(f64::NAN, 3.0), Err(SpecialError::NotANumber { .. })));
    assert!(matches!(chi2_sf(1.0, -2.0), Err(SpecialError::Domain { .. })));
    assert!(matches!(f_cdf(1.0, 0.0, 3.0), Err(SpecialError::Domain { .. })));
    assert!(matches!(ln_beta(1.0, 0.0), Err(SpecialError::Domain { .. })));
}

#[test]
fn ln_beta_matches_the_gamma_identity() {
    // B(a, b) = G(a) G(b) / G(a + b); B(1, 1) = 1; B(1/2, 1/2) = pi
    assert!(ln_beta(1.0, 1.0).unwrap().abs() < 1e-15);
    close("ln B(1/2, 1/2)", ln_beta(0.5, 0.5).unwrap(), std::f64::consts::PI.ln(), REL);
    // across the switch to the large-argument form
    for (a, b) in [(2.0, 30.0), (0.4, 15.0), (14.0, 15.0), (16.0, 400.0), (500.0, 0.5)] {
        let direct = ln_gamma(a).unwrap() + ln_gamma(b).unwrap() - ln_gamma(a + b).unwrap();
        let got = ln_beta(a, b).unwrap();
        assert!((got - direct).abs() <= 1e-11 * direct.abs().max(1.0), "ln_beta({a}, {b})");
    }
}
