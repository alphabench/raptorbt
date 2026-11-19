//! GARCH(1,1) and GJR-GARCH(1,1,1) in Rust against the reference fit.
//!
//! The reference numbers are arch 8.0.0's on the committed oracle series (the extract
//! `tests/oracle/stats_garch_extract.json`, cut from `stats_garch.json.gz` by
//! `tests/oracle/extract_stats_garch.py`). Bars: log-likelihood no lower than the reference's
//! minus 1e-3 (a higher one is a pass, listed with the numbers); where both fits reach the same
//! likelihood (within 1e-3) the parameters within 5e-3, the 10-bar forecast variance within 1e-3,
//! the conditional volatility within 1e-3 and the robust standard errors within 1e-3.
//!
//! The shape `nu` of the t is not held to the parameter bar: the t likelihood is flat in `nu`
//! where the data are close to normal, and it is checked through the likelihood and the forecast
//! it feeds. The numbers are printed (`cargo test --test stats_garch -- --nocapture`).

use std::collections::BTreeMap;

use raptorbt::stats::garch::{
    garch, log_likelihood_gradient, Dist, GarchError, GarchOptions, GarchResult, Model, SCALE,
};
use serde_json::Value;

const EXTRACT: &str = include_str!("oracle/stats_garch_extract.json");

fn f8s(v: &Value) -> Vec<f64> {
    let hex = v["f8"].as_str().expect("an {\"f8\": hex} array");
    (0..hex.len() / 16)
        .map(|i| f64::from_bits(u64::from_str_radix(&hex[16 * i..16 * i + 16], 16).unwrap()))
        .collect()
}

struct Case {
    name: String,
    returns: Vec<f64>,
    opts: GarchOptions,
    reference_error: Option<String>,
    lib: Option<Value>,
}

fn cases() -> Vec<Case> {
    let doc: Value = serde_json::from_str(EXTRACT).unwrap();
    let series: BTreeMap<String, Vec<f64>> =
        doc["series"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), f8s(v))).collect();
    doc["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            let kw = &c["kwargs"];
            let mut opts = GarchOptions::default();
            match kw["model"].as_str() {
                Some("gjr") => opts.model = Model::Gjr,
                _ => opts.model = Model::Garch,
            }
            if kw["dist"].as_str() == Some("normal") {
                opts.dist = Dist::Normal;
            }
            if let Some(h) = kw["horizon"].as_u64() {
                opts.horizon = h as usize;
            }
            opts.periods_per_year = kw["periods_per_year"].as_u64().map(|p| p as u32);
            Case {
                name: c["name"].as_str().unwrap().to_string(),
                returns: series[c["series"].as_str().unwrap()].clone(),
                opts,
                reference_error: c["reference_error"]["message"].as_str().map(str::to_string),
                lib: c.get("lib").cloned(),
            }
        })
        .collect()
}

fn case(name: &str) -> Case {
    cases().into_iter().find(|c| c.name == name).unwrap_or_else(|| panic!("no case {name}"))
}

/// |a - b| / max(|b|, floor)
fn rel(a: f64, b: f64, floor: f64) -> f64 {
    (a - b).abs() / b.abs().max(floor)
}

/// The fitted parameters in arch's units (percent) and order.
fn params_pct(r: &GarchResult) -> Vec<f64> {
    let p = &r.parameters;
    let mut v = vec![p.mu * SCALE, p.omega * SCALE * SCALE, p.alpha];
    v.extend(p.gamma);
    v.push(p.beta);
    v.extend(p.nu);
    v
}

#[test]
#[allow(clippy::needless_range_loop)]
fn replay_against_the_reference_fit() {
    let mut higher = Vec::new();
    let mut short = Vec::new();
    for c in cases() {
        let Some(lib) = &c.lib else { continue };
        if c.reference_error.is_some() || lib["convergence_flag"].as_i64() != Some(0) {
            continue; // refused by the reference, or a reference fit that did not converge
        }
        let res = garch(&c.returns, &c.opts).unwrap_or_else(|e| panic!("{}: {e}", c.name));
        let oracle_llf = f8s(&lib["loglikelihood"])[0];
        let d_llf = res.log_likelihood - oracle_llf;
        eprintln!("{:<30} dllf {:+.3e}  persistence {:.5}", c.name, d_llf, res.persistence);
        assert!(
            d_llf >= -1e-3,
            "{}: log-likelihood {} below the reference {}",
            c.name,
            res.log_likelihood,
            oracle_llf
        );
        if d_llf > 1e-3 {
            higher.push(format!("{} (+{:.4})", c.name, d_llf));
            continue; // another optimum at a higher likelihood: nothing else to compare
        }
        let mine = params_pct(&res);
        let theirs = f8s(&lib["params"]);
        assert_eq!(mine.len(), theirs.len(), "{}: parameter count", c.name);
        let names = &res.parameter_names;
        for (i, name) in names.iter().enumerate() {
            let err = match *name {
                "mu" | "omega" => rel(mine[i], theirs[i], 1e-2),
                "nu" => {
                    eprintln!(
                        "    nu {:.3} against {:.3} (not held to the bar)",
                        mine[i], theirs[i]
                    );
                    continue;
                }
                _ => (mine[i] - theirs[i]).abs(),
            };
            if err > 5e-3 {
                short.push(format!("{} {name}: {:.6} against {:.6}", c.name, mine[i], theirs[i]));
            }
        }
        // forecast variance and conditional volatility
        let fc = f8s(&lib["forecast_variance_10"]);
        for (h, v) in res.forecast_vol_per_period.iter().take(fc.len()).enumerate() {
            let e = rel(v * v * SCALE * SCALE, fc[h], 0.0);
            assert!(e <= 1e-3, "{}: forecast variance at h={} off by {e:e}", c.name, h + 1);
            // the variance the result reports, in decimal-return units: arch's percent^2 / 1e4
            let reported = res.forecast_variance_per_period[h] * SCALE * SCALE;
            let e = rel(reported, fc[h], 0.0);
            assert!(e <= 1e-3, "{}: reported variance at h={} off by {e:e}", c.name, h + 1);
        }
        let tail = f8s(&lib["conditional_volatility_tail"]);
        let ours = &res.recent_conditional_vol;
        for (a, b) in ours.iter().rev().zip(tail.iter().rev()) {
            let e = rel(a * SCALE, *b, 0.0);
            assert!(e <= 1e-3, "{}: conditional volatility off by {e:e}", c.name);
        }
        // the robust standard errors (the t shape is flat enough for them to differ a little)
        let cov =
            res.covariance_robust.as_ref().unwrap_or_else(|| panic!("{}: no covariance", c.name));
        let se = res.std_errors.as_ref().unwrap();
        assert_eq!(
            se.alpha.to_bits(),
            cov[2][2].sqrt().to_bits(),
            "{}: std error is the covariance diagonal",
            c.name
        );
        let k = names.len();
        for i in 0..k {
            for j in 0..i {
                assert_eq!(
                    cov[i][j].to_bits(),
                    cov[j][i].to_bits(),
                    "{}: covariance symmetric",
                    c.name
                );
            }
        }
        if c.opts.dist == Dist::Normal {
            let ref_cov = f8s(&lib["param_cov_robust"]);
            let unit = |i: usize| match names[i] {
                "mu" => SCALE,
                "omega" => SCALE * SCALE,
                _ => 1.0,
            };
            for i in 0..k {
                let e = rel(cov[i][i].sqrt() * unit(i), ref_cov[i * k + i].sqrt(), 0.0);
                assert!(e <= 1e-3, "{}: robust std error of {} off by {e:e}", c.name, names[i]);
            }
        }
    }
    eprintln!("higher likelihood than the reference: {higher:?}");
    assert!(short.is_empty(), "parameters beyond 5e-3 at the same likelihood: {short:?}");
}

#[test]
fn the_reference_refusals_and_their_messages() {
    let c = case("below_min_obs");
    let e = garch(&c.returns, &c.opts).unwrap_err();
    assert_eq!(e, GarchError::InsufficientObservations { found: 249, needed: 250 });
    assert_eq!(e.to_string(), c.reference_error.unwrap());
    assert_eq!(e.code(), "insufficient_observations");

    let c = case("constant");
    let e = garch(&c.returns, &c.opts).unwrap_err();
    assert_eq!(e, GarchError::DegenerateSeries);
    assert_eq!(e.to_string(), c.reference_error.unwrap());
    assert_eq!(e.code(), "degenerate_series");

    let c = case("bad_horizon");
    let e = garch(&c.returns, &c.opts).unwrap_err();
    assert_eq!(e, GarchError::BadHorizon { horizon: 61 });
    assert_eq!(e.to_string(), c.reference_error.unwrap());

    let mut with_nan = case("garch_normal").returns;
    with_nan[10] = f64::NAN;
    assert_eq!(garch(&with_nan, &GarchOptions::default()).unwrap_err(), GarchError::NonFinite);
}

/// The reference (arch) does not converge on returns of scale 1e-7; the fit rescales and answers,
/// and the answer is the one the same series at scale 1e-2 gives, in its own units.
#[test]
fn tiny_returns_fit_and_are_scale_equivariant() {
    let c = case("tiny_returns");
    assert!(c.reference_error.is_some(), "the reference did not converge on this series");
    let small = garch(&c.returns, &c.opts).expect("rescaled fit");
    assert_eq!(small.rescale, 1e5);
    let big_returns: Vec<f64> = c.returns.iter().map(|v| v * 1e5).collect();
    let big = garch(&big_returns, &c.opts).expect("fit at scale 1e-2");
    assert_eq!(big.rescale, 1.0);
    assert!((small.parameters.alpha - big.parameters.alpha).abs() < 1e-6);
    assert!((small.parameters.beta - big.parameters.beta).abs() < 1e-6);
    assert!(rel(small.parameters.omega * 1e10, big.parameters.omega, 0.0) < 1e-6);
    assert!(rel(small.parameters.mu * 1e5, big.parameters.mu, 1e-3) < 1e-6);
    // percent units differ by the factor: ll_small = ll_big + n ln(1e5)
    let shift = small.n as f64 * 1e5f64.ln();
    assert!(
        (small.log_likelihood - big.log_likelihood - shift).abs() < 1e-5,
        "{} {} {shift}",
        small.log_likelihood,
        big.log_likelihood
    );
    assert!(rel(small.last_vol_per_period * 1e5, big.last_vol_per_period, 0.0) < 1e-6);
}

/// Returns that differ by a few ulps of their level are the rounding noise, not a series: refused,
/// where the fit used to print an `omega` of 5e-35.
#[test]
fn returns_constant_to_a_few_ulps_are_refused() {
    let noisy: Vec<f64> =
        (0..500).map(|i| 0.001 * (1.0 + 1e-14 * ((i * 7 % 11) as f64 - 5.0))).collect();
    let e = garch(&noisy, &GarchOptions::default()).unwrap_err();
    assert_eq!(e, GarchError::DegenerateSeries);
    // the same spread around zero is not constant relative to its level
    let around_zero: Vec<f64> = (0..500).map(|i| 1e-14 * ((i * 7 % 11) as f64 - 5.0)).collect();
    assert_ne!(
        garch(&around_zero, &GarchOptions::default()).err(),
        Some(GarchError::DegenerateSeries)
    );
}

/// Persistence within 1e-6 of 1 is not a stationary variance: whatever the data, the flags agree
/// with the number and the half-life and unconditional volatility are absent exactly then.
#[test]
fn the_integrated_edge_is_flagged_and_has_no_half_life() {
    for name in ["garch_normal", "gjr_t"] {
        let c = case(name);
        let res = garch(&c.returns, &c.opts).unwrap();
        let edge = res.persistence >= 1.0 - 1e-6;
        assert_eq!(res.integrated, edge, "{name}");
        assert_eq!(res.stationary_variance, !edge, "{name}");
        assert_eq!(res.half_life_periods.is_none(), edge, "{name}");
        assert_eq!(res.unconditional_vol_per_period.is_none(), edge, "{name}");
        let at_bound = res.parameters.nu.is_some_and(|v| !(2.0501..=499.95).contains(&v));
        assert_eq!(res.shape_at_bound, at_bound, "{name}");
    }
}

fn lcg_returns(n: usize) -> Vec<f64> {
    // a fixed, plainly pseudo-random percent series with volatility clusters
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut vol = 1.0;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (state >> 11) as f64 / (1u64 << 53) as f64;
            let z = (u - 0.5) * 3.4;
            let r = vol * z;
            vol = 0.4 + 0.5 * vol + 0.3 * r.abs();
            r
        })
        .collect()
}

/// The analytic gradient against central differences of the likelihood, for each model and
/// distribution.
#[test]
fn analytic_gradient_matches_central_differences() {
    let y = lcg_returns(400);
    for (model, dist, params) in [
        (Model::Garch, Dist::Normal, vec![0.02, 0.05, 0.08, 0.88]),
        (Model::Garch, Dist::StudentT, vec![0.02, 0.05, 0.08, 0.88, 7.5]),
        (Model::Gjr, Dist::Normal, vec![-0.01, 0.04, 0.03, 0.09, 0.86]),
        (Model::Gjr, Dist::StudentT, vec![-0.01, 0.04, 0.03, 0.09, 0.86, 5.2]),
    ] {
        let (llf, grad) = log_likelihood_gradient(&y, model, dist, &params);
        assert!(llf.is_finite());
        for j in 0..params.len() {
            let h = 1e-5 * params[j].abs().max(1e-2);
            let mut up = params.clone();
            let mut dn = params.clone();
            up[j] += h;
            dn[j] -= h;
            let num = (log_likelihood_gradient(&y, model, dist, &up).0
                - log_likelihood_gradient(&y, model, dist, &dn).0)
                / (2.0 * h);
            let err = (grad[j] - num).abs() / grad[j].abs().max(1.0);
            assert!(
                err < 1e-5,
                "{model:?} {dist:?} parameter {j}: analytic {} against numeric {num}",
                grad[j]
            );
        }
    }
}

/// The normal GARCH(1,1) log-likelihood of three bars written out: backcast from the three
/// squared residuals about the mean with weights 1, 0.94, 0.94^2; `s_0 = w + (a + b) bc`;
/// `s_t = w + a e_{t-1}^2 + b s_{t-1}`; `ll = -(1/2) sum (ln 2 pi + ln s_t + e_t^2 / s_t)`.
#[test]
fn log_likelihood_of_three_bars_by_hand() {
    let y = [1.0, -2.0, 0.5];
    let (mu, w, a, b) = (0.1, 0.5, 0.1, 0.8);
    let mean = (1.0 - 2.0 + 0.5) / 3.0;
    let r: Vec<f64> = y.iter().map(|v| v - mean).collect();
    let (w0, w1, w2) = (1.0, 0.94, 0.94 * 0.94);
    let bc = (w0 * r[0] * r[0] + w1 * r[1] * r[1] + w2 * r[2] * r[2]) / (w0 + w1 + w2);
    let e: Vec<f64> = y.iter().map(|v| v - mu).collect();
    let s0 = w + (a + b) * bc;
    let s1 = w + a * e[0] * e[0] + b * s0;
    let s2 = w + a * e[1] * e[1] + b * s1;
    let ln2pi = (2.0 * std::f64::consts::PI).ln();
    let expected: f64 = [(s0, e[0]), (s1, e[1]), (s2, e[2])]
        .iter()
        .map(|(s, x)| -0.5 * (ln2pi + s.ln() + x * x / s))
        .sum();
    let (llf, _) = log_likelihood_gradient(&y, Model::Garch, Dist::Normal, &[mu, w, a, b]);
    assert!((llf - expected).abs() < 1e-12, "{llf} against {expected}");
}

/// The same input gives the same bits, whichever thread runs it.
#[test]
fn the_fit_is_bit_for_bit_repeatable() {
    let c = case("gjr_t");
    let a = garch(&c.returns, &c.opts).unwrap();
    let b = std::thread::spawn({
        let returns = c.returns.clone();
        let opts = c.opts;
        move || garch(&returns, &opts).unwrap()
    })
    .join()
    .unwrap();
    assert_eq!(a.log_likelihood.to_bits(), b.log_likelihood.to_bits());
    for (x, y) in params_pct(&a).iter().zip(params_pct(&b)) {
        assert_eq!(x.to_bits(), y.to_bits());
    }
    assert_eq!(a, b);
}

/// Results hold the keys' documented meaning: persistence, half-life and annualising.
#[test]
fn derived_fields() {
    let c = case("garch_horizon_60_annualised");
    let r = garch(&c.returns, &c.opts).unwrap();
    assert_eq!(r.forecast_vol_per_period.len(), 60);
    assert_eq!(r.recent_conditional_vol.len(), 60);
    assert_eq!(r.periods_per_year, Some(252));
    let ann = 252f64.sqrt();
    let fa = r.forecast_vol_annualised.as_ref().unwrap();
    for (a, p) in fa.iter().zip(&r.forecast_vol_per_period) {
        assert!((a - p * ann).abs() < 1e-15);
    }
    assert!((r.last_vol_annualised.unwrap() - r.last_vol_per_period * ann).abs() < 1e-15);
    let p = r.persistence;
    assert!((p - (r.parameters.alpha + r.parameters.beta)).abs() < 1e-15);
    assert!((r.half_life_periods.unwrap() - 0.5f64.ln() / p.ln()).abs() < 1e-12);
    let uncond = (r.parameters.omega / (1.0 - p)).sqrt();
    assert!((r.unconditional_vol_per_period.unwrap() - uncond).abs() < 1e-15);
    let cum: f64 = r.forecast_vol_per_period.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!((r.forecast_cumulative_vol - cum).abs() < 1e-15);
    assert_eq!(r.forecast_variance_per_period.len(), 60);
    for (v, s) in r.forecast_variance_per_period.iter().zip(&r.forecast_vol_per_period) {
        assert!((v - s * s).abs() <= 1e-15 * v, "variance is the square of the volatility");
    }
    let total: f64 = r.forecast_variance_per_period.iter().sum();
    assert_eq!(r.forecast_cumulative_variance, total);
    assert!((r.forecast_cumulative_vol.powi(2) - total).abs() <= 1e-12 * total);
    assert_eq!(r.model, "GARCH(1,1)");
    assert_eq!(r.aic, 2.0 * 4.0 - 2.0 * r.log_likelihood);
}
