use super::*;

fn inputs(right: OptionRight, s: f64, k: f64, tau: f64, vol: f64, r: f64, q: f64) -> OptionInputs {
    OptionInputs { right, underlying: s, strike: k, tau, vol, rate: r, dividend_yield: q }
}

const C: OptionRight = OptionRight::Call;
const P: OptionRight = OptionRight::Put;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + b.abs())
}

// Hull, Options Futures and Other Derivatives: S=42, K=40, r=10%, vol=20%,
// T=0.5 gives call 4.759 and put 0.808.
#[test]
fn textbook_values() {
    let c = price(Model::BlackScholes, &inputs(C, 42.0, 40.0, 0.5, 0.2, 0.1, 0.0)).unwrap();
    let p = price(Model::BlackScholes, &inputs(P, 42.0, 40.0, 0.5, 0.2, 0.1, 0.0)).unwrap();
    assert!((c - 4.759422).abs() < 1e-5, "{c}");
    assert!((p - 0.808599).abs() < 1e-5, "{p}");
}

#[test]
fn put_call_parity_with_carry() {
    for &(r, q) in &[(0.0, 0.0), (0.06, 0.015), (-0.01, 0.03)] {
        for &k in &[70.0, 100.0, 140.0] {
            for &tau in &[0.01, 0.5, 3.0] {
                let c = inputs(C, 100.0, k, tau, 0.3, r, q);
                let p = OptionInputs { right: P, ..c };
                let lhs = price_raw(Model::BlackScholes, &c) - price_raw(Model::BlackScholes, &p);
                let rhs = 100.0 * (-q * tau).exp() - k * (-r * tau).exp();
                assert!(close(lhs, rhs, 1e-12), "{lhs} {rhs}");
            }
        }
    }
}

#[test]
fn black76_matches_black_scholes_on_the_forward() {
    let (s, r, q, tau) = (100.0, 0.05, 0.02, 0.75);
    let f = forward_price(s, tau, r, q);
    for right in [C, P] {
        let bs = greeks_raw(Model::BlackScholes, &inputs(right, s, 95.0, tau, 0.25, r, q));
        let b76 = greeks_raw(Model::Black76, &inputs(right, f, 95.0, tau, 0.25, r, 0.0));
        assert!(close(bs.price, b76.price, 1e-13));
        assert!(close(bs.vega, b76.vega, 1e-12));
        // Black-76 delta is per unit of forward.
        assert!(close(bs.delta, b76.delta * ((r - q) * tau).exp(), 1e-12));
    }
}

fn bump(i: &OptionInputs, ds: f64, dv: f64, dt: f64, dr: f64) -> OptionInputs {
    OptionInputs {
        underlying: i.underlying + ds,
        vol: i.vol + dv,
        tau: i.tau + dt,
        rate: i.rate + dr,
        ..*i
    }
}

#[test]
fn greeks_match_central_differences() {
    for model in [Model::BlackScholes, Model::Black76] {
        for right in [C, P] {
            for &(s, k, tau, vol, r) in &[
                (100.0, 100.0, 0.5, 0.2, 0.05),
                (100.0, 80.0, 1.0, 0.35, 0.03),
                (100.0, 130.0, 0.1, 0.5, -0.01),
            ] {
                let q = if model == Model::Black76 { 0.0 } else { 0.02 };
                let i = inputs(right, s, k, tau, vol, r, q);
                let g = greeks_raw(model, &i);
                let p = |x: &OptionInputs| price_raw(model, x);
                let (hs, hv, ht, hr) = (1e-3, 1e-5, 1e-6, 1e-6);
                let delta = (p(&bump(&i, hs, 0.0, 0.0, 0.0)) - p(&bump(&i, -hs, 0.0, 0.0, 0.0)))
                    / (2.0 * hs);
                let gamma = (p(&bump(&i, hs, 0.0, 0.0, 0.0)) - 2.0 * g.price
                    + p(&bump(&i, -hs, 0.0, 0.0, 0.0)))
                    / (hs * hs);
                let vega = (p(&bump(&i, 0.0, hv, 0.0, 0.0)) - p(&bump(&i, 0.0, -hv, 0.0, 0.0)))
                    / (2.0 * hv);
                // Theta is minus the derivative in tau; for Black-76 the
                // forward is held fixed, as the formula assumes.
                let theta = -(p(&bump(&i, 0.0, 0.0, ht, 0.0)) - p(&bump(&i, 0.0, 0.0, -ht, 0.0)))
                    / (2.0 * ht);
                let rho = (p(&bump(&i, 0.0, 0.0, 0.0, hr)) - p(&bump(&i, 0.0, 0.0, 0.0, -hr)))
                    / (2.0 * hr);
                let what = format!("{model:?} {right:?} {s} {k} {tau} {vol}");
                assert!(close(g.delta, delta, 1e-6), "delta {what}: {} {delta}", g.delta);
                assert!(close(g.gamma, gamma, 1e-4), "gamma {what}: {} {gamma}", g.gamma);
                assert!(close(g.vega, vega, 1e-6), "vega {what}");
                assert!(close(g.theta, theta, 1e-5), "theta {what}: {} {theta}", g.theta);
                if model == Model::BlackScholes {
                    assert!(close(g.rho, rho, 1e-5), "rho {what}: {} {rho}", g.rho);
                }
            }
        }
    }
}

#[test]
fn expiry_is_intrinsic() {
    let c = greeks_raw(Model::BlackScholes, &inputs(C, 105.0, 100.0, 0.0, 0.3, 0.05, 0.0));
    assert_eq!(c, Greeks { price: 5.0, delta: 1.0, gamma: 0.0, vega: 0.0, theta: 0.0, rho: 0.0 });
    let p = greeks_raw(Model::BlackScholes, &inputs(P, 105.0, 100.0, 0.0, 0.3, 0.05, 0.0));
    assert_eq!((p.price, p.delta), (0.0, 0.0));
    let p = greeks_raw(Model::BlackScholes, &inputs(P, 95.0, 100.0, 0.0, 0.3, 0.05, 0.0));
    assert_eq!((p.price, p.delta), (5.0, -1.0));
}

#[test]
fn zero_vol_is_the_discounted_forward_payoff() {
    let (s, k, tau, r, q) = (100.0, 98.0, 0.5, 0.06, 0.01);
    let c = greeks_raw(Model::BlackScholes, &inputs(C, s, k, tau, 0.0, r, q));
    let f = forward_price(s, tau, r, q);
    assert!(close(c.price, (-r * tau).exp() * (f - k), 1e-14));
    assert!(close(c.delta, (-q * tau).exp(), 1e-14));
    assert_eq!((c.gamma, c.vega), (0.0, 0.0));
    // Out of the money on the forward: worth nothing, no delta.
    let o = greeks_raw(Model::BlackScholes, &inputs(C, s, 120.0, tau, 0.0, r, q));
    assert_eq!((o.price, o.delta, o.theta, o.rho), (0.0, 0.0, 0.0, 0.0));
    // Delta follows the forward, not the spot: S < K < F.
    let itm_on_fwd = greeks_raw(Model::BlackScholes, &inputs(C, 100.0, 102.0, 1.0, 0.0, 0.08, 0.0));
    assert!(itm_on_fwd.delta > 0.9);
}

#[test]
fn invalid_inputs_are_errors_or_nan_never_zero() {
    let bad = [
        inputs(C, 0.0, 100.0, 1.0, 0.2, 0.0, 0.0),
        inputs(C, 100.0, -1.0, 1.0, 0.2, 0.0, 0.0),
        inputs(C, 100.0, 100.0, -0.1, 0.2, 0.0, 0.0),
        inputs(C, 100.0, 100.0, 1.0, -0.2, 0.0, 0.0),
        inputs(C, f64::NAN, 100.0, 1.0, 0.2, 0.0, 0.0),
        inputs(C, 100.0, 100.0, 1.0, f64::INFINITY, 0.0, 0.0),
        inputs(C, 100.0, 100.0, 1.0, 0.2, f64::NAN, 0.0),
    ];
    for i in bad {
        assert!(price(Model::BlackScholes, &i).is_err(), "{i:?}");
        assert!(price_or_nan(Model::BlackScholes, &i).is_nan(), "{i:?}");
        assert!(greeks_or_nan(Model::BlackScholes, &i).delta.is_nan());
    }
    // Black-76 refuses a dividend yield instead of ignoring it.
    assert!(price(Model::Black76, &inputs(C, 100.0, 100.0, 1.0, 0.2, 0.0, 0.01)).is_err());
}

#[test]
fn extreme_inputs_stay_finite_and_bounded() {
    for &tau in &[1e-9, 1e-6, 10.0, 50.0] {
        for &vol in &[1e-6, 0.01, 5.0, 20.0] {
            for &k in &[1.0, 100.0, 1e4] {
                let g = greeks_raw(Model::BlackScholes, &inputs(C, 100.0, k, tau, vol, 0.05, 0.0));
                assert!(g.price.is_finite() && g.price >= 0.0 && g.price <= 100.0 + 1e-9, "{g:?}");
                assert!((0.0..=1.0 + 1e-12).contains(&g.delta), "{g:?}");
                assert!(g.gamma.is_finite() && g.vega.is_finite() && g.theta.is_finite(), "{g:?}");
            }
        }
    }
}

#[test]
fn implied_vol_round_trips() {
    let search = IvSearch::default();
    for model in [Model::BlackScholes, Model::Black76] {
        for right in [C, P] {
            for &k in &[60.0, 90.0, 100.0, 110.0, 150.0] {
                for &tau in &[2.0 / 365.0, 0.25, 2.0] {
                    for &vol in &[0.02, 0.15, 0.6, 3.0] {
                        let q = if model == Model::Black76 { 0.0 } else { 0.01 };
                        let i = inputs(right, 100.0, k, tau, vol, 0.04, q);
                        let target = price_raw(model, &i);
                        let (iv, status) = implied_vol_raw(model, &i, target, &search);
                        if status != IvStatus::Ok {
                            // Only a price with no time value left (deep
                            // in the money) may fail, and it says so.
                            assert!(
                                matches!(
                                    status,
                                    IvStatus::BelowBracket
                                        | IvStatus::AboveUpperBound
                                        | IvStatus::BelowIntrinsic
                                ),
                                "{model:?} {right:?} {k} {tau} {vol}: {status:?}"
                            );
                            continue;
                        }
                        let back = price_raw(model, &OptionInputs { vol: iv, ..i });
                        assert!(
                            (back - target).abs() <= 1e-12 * (1.0 + target),
                            "{model:?} {right:?} {k} {tau} {vol}: {back} vs {target}"
                        );
                        let vega = greeks_raw(model, &i).vega;
                        if vega > 1e-3 {
                            assert!((iv - vol).abs() < 1e-8, "{iv} vs {vol}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn implied_vol_reports_why_there_is_none() {
    let s = IvSearch::default();
    let base = inputs(C, 100.0, 90.0, 0.25, 0.0, 0.0, 0.0);
    let status = |t: f64| implied_vol_raw(Model::BlackScholes, &base, t, &s).1;
    assert_eq!(status(5.0), IvStatus::BelowIntrinsic); // intrinsic is 10
    assert_eq!(status(100.0), IvStatus::AboveUpperBound);
    assert_eq!(status(150.0), IvStatus::AboveUpperBound);
    assert_eq!(status(10.0), IvStatus::BelowBracket); // intrinsic, no time value
    assert_eq!(status(99.9), IvStatus::AboveBracket); // needs vol far above 500%
    assert_eq!(status(0.0), IvStatus::BelowIntrinsic);
    assert_eq!(status(-1.0), IvStatus::BelowIntrinsic);
    let otm = inputs(C, 100.0, 150.0, 0.25, 0.0, 0.0, 0.0);
    assert_eq!(implied_vol_raw(Model::BlackScholes, &otm, 0.0, &s).1, IvStatus::BelowBracket);
    assert_eq!(status(f64::NAN), IvStatus::InvalidInput);
    let expired = OptionInputs { tau: 0.0, ..base };
    assert_eq!(implied_vol_raw(Model::BlackScholes, &expired, 10.0, &s).1, IvStatus::Expired);
    let (iv, st) = implied_vol_raw(Model::BlackScholes, &base, 5.0, &s);
    assert!(iv.is_nan() && st != IvStatus::Ok, "no answer is NaN, never zero");
    assert_eq!(implied_vol(Model::BlackScholes, &base, 5.0, &s), Err(IvStatus::BelowIntrinsic));
    assert!(IvSearch { lo: 0.0, ..s }.validate().is_err());
    assert!(IvSearch { hi: 1e-5, ..s }.validate().is_err());
}

#[test]
fn iterations_budget_is_reported() {
    let base = inputs(C, 100.0, 100.0, 0.25, 0.0, 0.0, 0.0);
    let target = price_raw(Model::BlackScholes, &OptionInputs { vol: 0.37, ..base });
    let tight = IvSearch { max_iter: 1, tol: 1e-14, ..IvSearch::default() };
    assert_eq!(
        implied_vol_raw(Model::BlackScholes, &base, target, &tight).1,
        IvStatus::NoConvergence
    );
}

#[test]
fn status_codes_round_trip() {
    for code in 0..=7u8 {
        let st = IvStatus::from_code(code).unwrap();
        assert_eq!(st as u8, code);
        assert!(!st.as_str().is_empty());
    }
    assert!(IvStatus::from_code(8).is_none());
}

#[test]
fn model_names() {
    assert_eq!(Model::parse("black76").unwrap(), Model::Black76);
    assert_eq!(Model::parse("black_scholes").unwrap().as_str(), "black_scholes");
    assert!(Model::parse("heston").is_err());
}
