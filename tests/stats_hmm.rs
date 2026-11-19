//! Hidden Markov regimes in Rust against statsmodels' `MarkovRegression`.
//!
//! The reference numbers are statsmodels 0.15.0's on the committed oracle series (the extract
//! `tests/oracle/stats_hmm_extract.json`, cut from `stats_hmm.json.gz` by
//! `tests/oracle/extract_stats_hmm.py`). Bars: log-likelihood no lower than the reference's minus
//! 1e-4 (a higher one, or another local optimum at a higher likelihood, is a pass, listed with
//! the numbers); where the two reach the same likelihood (within 1e-6) the parameters and the
//! probabilities within 1e-3, after ordering both by variance so that a swap of labels is not a
//! difference. The numbers are printed (`cargo test --test stats_hmm -- --nocapture`).

use std::collections::BTreeMap;

use raptorbt::stats::hmm::{
    filter_and_smooth, markov_regimes, mean_nll_gradient, HmmError, HmmOptions, HmmResult, SCALE,
    SMOOTHED_NOTE,
};
use serde_json::Value;

const EXTRACT: &str = include_str!("oracle/stats_hmm_extract.json");

/// Every float under `v` ({"f8": hex} arrays), in order.
fn flat(v: &Value) -> Vec<f64> {
    match v {
        Value::Object(o) if o.contains_key("f8") => {
            let hex = o["f8"].as_str().unwrap();
            (0..hex.len() / 16)
                .map(|i| {
                    f64::from_bits(u64::from_str_radix(&hex[16 * i..16 * i + 16], 16).unwrap())
                })
                .collect()
        }
        Value::Array(a) => a.iter().flat_map(flat).collect(),
        other => panic!("not a float array: {other}"),
    }
}

fn scalar(v: &Value) -> f64 {
    flat(v)[0]
}

struct Case {
    name: String,
    returns: Vec<f64>,
    regimes: usize,
    reference_error: Option<String>,
    reference: Option<Value>,
    lib: Option<Value>,
}

fn cases() -> Vec<Case> {
    let doc: Value = serde_json::from_str(EXTRACT).unwrap();
    let series: BTreeMap<String, Vec<f64>> =
        doc["series"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), flat(v))).collect();
    doc["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| Case {
            name: c["name"].as_str().unwrap().to_string(),
            returns: series[c["series"].as_str().unwrap()].clone(),
            regimes: c["kwargs"]["k"].as_u64().unwrap() as usize,
            reference_error: c["reference_error"]["message"].as_str().map(str::to_string),
            reference: c.get("reference").cloned(),
            lib: c.get("lib").cloned(),
        })
        .collect()
}

fn case(name: &str) -> Case {
    cases().into_iter().find(|c| c.name == name).unwrap_or_else(|| panic!("no case {name}"))
}

fn fit(c: &Case) -> Result<HmmResult, HmmError> {
    markov_regimes(&c.returns, &HmmOptions { regimes: c.regimes, ..HmmOptions::default() })
}

fn max_abs(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}

/// |a - b| / max(1, |b|)
fn near(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1.0)
}

/// The reference's fit in regimes ordered calmest first: (mean, variance, `trans[from][to]`).
fn reference_ordered(lib: &Value, k: usize) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let params = flat(&lib["params"]);
    let to_from = flat(&lib["regime_transition"]); // [to][from]
    let mu = &params[k * (k - 1)..k * (k - 1) + k];
    let var = &params[k * (k - 1) + k..];
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|a, b| var[*a].total_cmp(&var[*b]));
    (
        order.iter().map(|o| mu[*o]).collect(),
        order.iter().map(|o| var[*o]).collect(),
        order.iter().map(|f| order.iter().map(|t| to_from[t * k + f]).collect()).collect(),
    )
}

#[test]
fn replay_against_the_reference_fit() {
    let mut differences = Vec::new();
    for c in cases() {
        let (Some(lib), Some(want)) = (&c.lib, &c.reference) else { continue };
        let k = c.regimes;
        let res = fit(&c).unwrap_or_else(|e| panic!("{}: {e}", c.name));
        let oracle_llf = scalar(&lib["llf"]);
        let d_llf = res.log_likelihood - oracle_llf;
        eprintln!(
            "{:<26} dllf {:+.3e}  starts {:?}  winner {}",
            c.name,
            d_llf,
            res.start_log_likelihoods
                .iter()
                .map(|l| l.map(|v| (v * 1e4).round() / 1e4))
                .collect::<Vec<_>>(),
            res.winning_start
        );
        assert!(
            d_llf >= -1e-4,
            "{}: log-likelihood {} below the reference {}",
            c.name,
            res.log_likelihood,
            oracle_llf
        );
        assert_eq!(res.aic, 2.0 * (k * (k - 1) + 2 * k) as f64 - 2.0 * res.log_likelihood);

        // regimes come out calmest first, whatever order the fit found them in
        for w in res.regime_stats.windows(2) {
            assert!(w[0].vol_per_period <= w[1].vol_per_period, "{}: regimes not ordered", c.name);
        }

        let (mu, var, trans) = reference_ordered(lib, k);
        let mine_mu: Vec<f64> =
            res.regime_stats.iter().map(|r| r.mean_per_period * SCALE).collect();
        let mine_var: Vec<f64> =
            res.regime_stats.iter().map(|r| (r.vol_per_period * SCALE).powi(2)).collect();
        let tail = |key: &str| flat(&lib[key]);
        let rows = tail("smoothed_marginal_probabilities_tail").len() / k;
        let ref_smoothed = tail("smoothed_marginal_probabilities_tail");
        let ref_filtered = tail("filtered_marginal_probabilities_tail");
        // the reference's probability columns in calmest-first order
        let order: Vec<usize> = {
            let params = flat(&lib["params"]);
            let v = &params[k * (k - 1) + k..];
            let mut o: Vec<usize> = (0..k).collect();
            o.sort_by(|a, b| v[*a].total_cmp(&v[*b]));
            o
        };
        let ordered = |m: &[f64]| -> Vec<f64> {
            (0..rows).flat_map(|t| order.iter().map(move |o| m[t * k + o])).collect()
        };
        let n = res.n;
        let mine_rows = |m: &Vec<Vec<f64>>| -> Vec<f64> {
            m[n - rows..].iter().flat_map(|r| r.iter().copied()).collect()
        };
        let d_mu = (0..k).map(|i| near(mine_mu[i], mu[i])).fold(0.0, f64::max);
        let d_var = (0..k).map(|i| near(mine_var[i], var[i])).fold(0.0, f64::max);
        let d_trans =
            (0..k).map(|f| max_abs(&res.transition_matrix[f], &trans[f])).fold(0.0, f64::max);
        let d_smoothed = max_abs(&mine_rows(&res.smoothed_probabilities), &ordered(&ref_smoothed));
        let d_filtered = max_abs(&mine_rows(&res.filtered_probabilities), &ordered(&ref_filtered));
        eprintln!(
            "    mean {d_mu:.2e}  variance {d_var:.2e}  transition {d_trans:.2e}  smoothed {d_smoothed:.2e}  filtered {d_filtered:.2e}"
        );
        if d_llf.abs() < 1e-6 {
            for (what, d) in [
                ("mean", d_mu),
                ("variance", d_var),
                ("transition", d_trans),
                ("smoothed probabilities", d_smoothed),
                ("filtered probabilities", d_filtered),
            ] {
                assert!(d <= 1e-3, "{}: {what} off by {d:e} at the same likelihood", c.name);
            }
            // the reference's answer, at full precision
            let want_stats = want["regime_stats"].as_array().unwrap();
            for (i, r) in res.regime_stats.iter().enumerate() {
                assert!(
                    near(
                        r.mean_per_period * SCALE,
                        scalar(&want_stats[i]["mean_per_period"]) * SCALE
                    ) <= 1e-3
                );
                assert!(
                    near(
                        r.vol_per_period * SCALE,
                        scalar(&want_stats[i]["vol_per_period"]) * SCALE
                    ) <= 1e-3
                );
                assert!(
                    (r.share_of_periods - scalar(&want_stats[i]["share_of_periods"])).abs() <= 1e-3,
                    "{}: share of regime {i}",
                    c.name
                );
            }
            assert_eq!(
                res.current_regime as u64,
                want["current_regime"].as_u64().unwrap(),
                "{}",
                c.name
            );
            assert_eq!(
                res.timeline_total as u64,
                want["timeline_total"].as_u64().unwrap(),
                "{}",
                c.name
            );
            let want_timeline = want["timeline"].as_array().unwrap();
            assert_eq!(res.timeline.len(), want_timeline.len(), "{}: timeline length", c.name);
            for (a, b) in res.timeline.iter().zip(want_timeline) {
                assert_eq!(a.regime as u64, b["regime"].as_u64().unwrap(), "{}", c.name);
                assert_eq!(a.start_index as u64, b["start_index"].as_u64().unwrap(), "{}", c.name);
                assert_eq!(a.end_index as u64, b["end_index"].as_u64().unwrap(), "{}", c.name);
            }
            let want_values = flat(&want["transition_matrix"]["values"]);
            let mine_values: Vec<f64> = res.transition_matrix.iter().flatten().copied().collect();
            assert!(max_abs(&mine_values, &want_values) <= 1e-3, "{}: transition matrix", c.name);
            let want_recent = flat(&want["recent_probabilities"]);
            let mine_recent: Vec<f64> =
                res.recent_probabilities.iter().flatten().copied().collect();
            assert!(
                max_abs(&mine_recent, &want_recent) <= 1e-3,
                "{}: recent probabilities",
                c.name
            );
        } else if d_llf > 1e-6 {
            differences.push(format!("{} (+{d_llf:.2e})", c.name));
        }
    }
    eprintln!("not at the reference's optimum to 1e-6 (every one at or above its likelihood): {differences:?}");
}

#[test]
fn the_reference_refusals_and_their_messages() {
    for (name, found, needed) in [("below_min_obs", 199, 200), ("k3_below_min_obs", 399, 400)] {
        let c = case(name);
        let e = fit(&c).unwrap_err();
        assert_eq!(e, HmmError::InsufficientObservations { found, needed });
        assert_eq!(e.to_string(), c.reference_error.unwrap());
        assert_eq!(e.code(), "insufficient_observations");
    }
    let c = case("constant");
    let e = fit(&c).unwrap_err();
    assert_eq!(e, HmmError::DegenerateSeries);
    assert_eq!(e.to_string(), c.reference_error.unwrap());
    assert_eq!(e.code(), "degenerate_series");

    let c = case("bad_k");
    let e = fit(&c).unwrap_err();
    assert_eq!(e, HmmError::BadRegimes { regimes: 4 });
    assert_eq!(e.to_string(), c.reference_error.unwrap());

    let mut with_nan = case("two_regimes").returns;
    with_nan[3] = f64::INFINITY;
    assert_eq!(markov_regimes(&with_nan, &HmmOptions::default()).unwrap_err(), HmmError::NonFinite);
}

/// The smoothed probabilities say they are not causal; the filtered ones are; the two meet at the
/// last bar; every row is a distribution.
#[test]
fn smoothed_is_marked_non_causal_and_filtered_is_causal() {
    let c = case("two_regimes");
    let res = fit(&c).unwrap();
    assert!(!res.smoothed_is_causal);
    assert!(res.filtered_is_causal);
    assert!(SMOOTHED_NOTE.contains("not causal"));
    assert_eq!(res.smoothed_probabilities.len(), res.n);
    assert_eq!(res.filtered_probabilities.len(), res.n);
    for row in res.smoothed_probabilities.iter().chain(&res.filtered_probabilities) {
        assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
    let (s, f) = (&res.smoothed_probabilities[res.n - 1], &res.filtered_probabilities[res.n - 1]);
    assert!(max_abs(s, f) < 1e-12, "smoothed and filtered agree at the last bar");
    assert_eq!(res.current_probabilities, *s);
    assert_eq!(res.recent_probabilities.len(), 20);
    assert_eq!(res.recent_filtered_probabilities.len(), 20);
    // and elsewhere they differ: the smoothed bar uses the future
    let gap = (0..res.n - 1)
        .map(|t| max_abs(&res.smoothed_probabilities[t], &res.filtered_probabilities[t]))
        .fold(0.0, f64::max);
    assert!(gap > 1e-3, "smoothed and filtered never differ ({gap:e})");
}

/// Changing returns after bar `t` leaves the filtered probabilities up to `t` bit for bit as they
/// were, and moves the smoothed one at `t`.
#[test]
fn filtered_probabilities_do_not_see_the_future() {
    let c = case("two_regimes");
    let res = fit(&c).unwrap();
    let y: Vec<f64> = c.returns.iter().map(|v| v * SCALE).collect();
    let mean: Vec<f64> = res.regime_stats.iter().map(|r| r.mean_per_period * SCALE).collect();
    let var: Vec<f64> =
        res.regime_stats.iter().map(|r| (r.vol_per_period * SCALE).powi(2)).collect();
    let (f0, s0, ll0) = filter_and_smooth(&y, &res.transition_matrix, &mean, &var).unwrap();
    let cut = 100;
    let mut changed = y.clone();
    for v in changed[cut + 1..].iter_mut() {
        *v = *v * 3.0 + 1.5;
    }
    let (f1, s1, ll1) = filter_and_smooth(&changed, &res.transition_matrix, &mean, &var).unwrap();
    for t in 0..=cut {
        for (a, b) in f0[t].iter().zip(&f1[t]) {
            assert_eq!(a.to_bits(), b.to_bits(), "filtered bar {t} moved with the future");
        }
    }
    assert!(max_abs(&s0[cut], &s1[cut]) > 1e-6 || max_abs(&s0[cut - 1], &s1[cut - 1]) > 1e-6);
    assert_ne!(ll0.to_bits(), ll1.to_bits());
}

/// The filter and smoother against the sum over every regime path of three bars.
#[test]
fn filter_and_smoother_match_enumeration_of_paths() {
    // transition[from][to]; the steady state of this chain is (2/3, 1/3)
    let transition = vec![vec![0.9, 0.1], vec![0.2, 0.8]];
    let pi = [2.0 / 3.0, 1.0 / 3.0];
    let mean = [0.0f64, 3.0];
    let var = [1.0f64, 2.0];
    let y = [0.1f64, 2.5, 0.4];
    let dens = |t: usize, s: usize| {
        (-(y[t] - mean[s]).powi(2) / (2.0 * var[s])).exp()
            / (2.0 * std::f64::consts::PI * var[s]).sqrt()
    };
    // joint of (s0, s1, s2) and the data
    let mut joint = [[[0.0f64; 2]; 2]; 2];
    for s0 in 0..2 {
        for s1 in 0..2 {
            for s2 in 0..2 {
                joint[s0][s1][s2] = pi[s0]
                    * dens(0, s0)
                    * transition[s0][s1]
                    * dens(1, s1)
                    * transition[s1][s2]
                    * dens(2, s2);
            }
        }
    }
    let total: f64 = joint.iter().flatten().flatten().sum();
    let mut smoothed = [[0.0f64; 2]; 3];
    for s0 in 0..2 {
        for s1 in 0..2 {
            for s2 in 0..2 {
                let p = joint[s0][s1][s2] / total;
                smoothed[0][s0] += p;
                smoothed[1][s1] += p;
                smoothed[2][s2] += p;
            }
        }
    }
    // filtered: the same sum over the prefix only
    let mut filtered = [[0.0f64; 2]; 3];
    for s0 in 0..2 {
        filtered[0][s0] = pi[s0] * dens(0, s0);
    }
    for s0 in 0..2 {
        for s1 in 0..2 {
            filtered[1][s1] += pi[s0] * dens(0, s0) * transition[s0][s1] * dens(1, s1);
            for s2 in 0..2 {
                filtered[2][s2] += pi[s0]
                    * dens(0, s0)
                    * transition[s0][s1]
                    * dens(1, s1)
                    * transition[s1][s2]
                    * dens(2, s2);
            }
        }
    }
    for row in filtered.iter_mut() {
        let s: f64 = row.iter().sum();
        row.iter_mut().for_each(|v| *v /= s);
    }
    let (f, s, ll) = filter_and_smooth(&y, &transition, &mean, &var).unwrap();
    assert!((ll - total.ln()).abs() < 1e-12, "{ll} against {}", total.ln());
    for t in 0..3 {
        assert!(max_abs(&f[t], &filtered[t]) < 1e-12, "filtered bar {t}");
        assert!(max_abs(&s[t], &smoothed[t]) < 1e-12, "smoothed bar {t}");
    }
}

/// The analytic gradient of the likelihood against central differences.
#[test]
fn analytic_gradient_matches_central_differences() {
    let c = case("three_regimes");
    let y: Vec<f64> = c.returns[..400].iter().map(|v| v * SCALE).collect();
    for (k, coords) in [
        (2usize, vec![2.0, 1.5, 0.2, -0.3, 0.3f64.ln(), 3.0f64.ln()]),
        (
            3,
            vec![
                1.2,
                0.8,
                1.0,
                0.1,
                0.4,
                -0.2,
                0.2,
                0.0,
                -0.3,
                0.1f64.ln(),
                1.0f64.ln(),
                5.0f64.ln(),
            ],
        ),
    ] {
        let (f, grad) = mean_nll_gradient(&y, &coords, k).unwrap();
        assert!(f.is_finite());
        for j in 0..coords.len() {
            let h = 1e-6;
            let mut up = coords.clone();
            let mut dn = coords.clone();
            up[j] += h;
            dn[j] -= h;
            let num = (mean_nll_gradient(&y, &up, k).unwrap().0
                - mean_nll_gradient(&y, &dn, k).unwrap().0)
                / (2.0 * h);
            assert!(
                (grad[j] - num).abs() < 1e-7,
                "k={k} coordinate {j}: analytic {} against numeric {num}",
                grad[j]
            );
        }
    }
}

/// The same input gives the same bits, whichever thread runs it.
#[test]
fn the_fit_is_bit_for_bit_repeatable() {
    let c = case("three_regimes");
    let a = fit(&c).unwrap();
    let returns = c.returns.clone();
    let regimes = c.regimes;
    let b = std::thread::spawn(move || {
        markov_regimes(&returns, &HmmOptions { regimes, ..HmmOptions::default() }).unwrap()
    })
    .join()
    .unwrap();
    assert_eq!(a.log_likelihood.to_bits(), b.log_likelihood.to_bits());
    assert_eq!(a, b);
}

/// The support flag: a regime of a handful of bars is flagged, the others are not.
#[test]
fn a_regime_on_a_few_bars_is_flagged_low_support() {
    let c = case("single_regime_data");
    let res = fit(&c).unwrap();
    for r in &res.regime_stats {
        let few = r.support_bars < 10.0;
        let short = r.expected_duration_periods.is_some_and(|d| d < 2.0);
        assert_eq!(r.low_support, few || short, "regime {}", r.regime);
    }
    let flagged: Vec<usize> =
        res.regime_stats.iter().filter(|r| r.low_support).map(|r| r.regime).collect();
    assert_eq!(res.low_support_regimes, flagged);
    let sum: f64 = res.regime_stats.iter().map(|r| r.support_bars).sum();
    assert!((sum - res.n as f64).abs() < 1e-9);
    // the option is a parameter
    let all = markov_regimes(
        &c.returns,
        &HmmOptions { min_support_bars: res.n as f64, ..HmmOptions::default() },
    )
    .unwrap();
    assert_eq!(all.low_support_regimes.len(), all.regimes);
}

/// Returns that differ by a few ulps of their level are refused, as GARCH's are.
#[test]
fn returns_constant_to_a_few_ulps_are_refused() {
    let noisy: Vec<f64> =
        (0..500).map(|i| 0.001 * (1.0 + 1e-14 * ((i * 7 % 11) as f64 - 5.0))).collect();
    let e = markov_regimes(&noisy, &HmmOptions::default()).unwrap_err();
    assert_eq!(e, HmmError::DegenerateSeries);
    assert_eq!(e.code(), "degenerate_series");
}

/// The fit does not depend on the units of the returns: at 1e-7 it used to fail to converge. The
/// series is rescaled by a power of ten and every result comes back in the units given.
#[test]
fn the_fit_does_not_depend_on_the_units_of_the_returns() {
    let c = case("two_regimes");
    let base = fit(&c).unwrap();
    assert_eq!(base.rescale, 1.0);
    for s in [1e-7, 1e-9, 1e3] {
        let small: Vec<f64> = c.returns.iter().map(|v| v * s).collect();
        let got =
            markov_regimes(&small, &HmmOptions { regimes: c.regimes, ..HmmOptions::default() })
                .unwrap_or_else(|e| panic!("scale {s}: {e}"));
        assert_ne!(got.rescale, 1.0, "scale {s}");
        for (a, b) in got.regime_stats.iter().zip(&base.regime_stats) {
            assert!((a.vol_per_period / s / b.vol_per_period - 1.0).abs() < 1e-4, "scale {s}");
            assert!((a.stay_probability - b.stay_probability).abs() < 1e-4, "scale {s}");
        }
        // the likelihood of the percent series moves by -n ln(s)
        let shift = -(base.n as f64) * s.ln();
        assert!(
            (got.log_likelihood - base.log_likelihood - shift).abs() < 1e-3 * shift.abs().max(1.0)
        );
    }
}
