//! Engle-Granger and Johansen replayed against statsmodels (`tests/python/oracle/fixtures/stats/
//! stats_coint.json.gz`, cut into `tests/oracle/stats_coint_extract.json` by
//! `tests/oracle/extract_stats_stationarity_coint.py`).
//!
//! Bars: Engle-Granger statistic, p-value, regression and spread 1e-8 (absolute below 1,
//! relative above); Johansen eigenvalues and statistics 1e-8; eigenvectors 1e-6 of the largest
//! entry of their column after the sign rule of `coint.rs` (the oracle's columns are signed by
//! the same rule here, since LAPACK's are arbitrary); critical values from the tables exactly.
//!
//! Panels the oracle answers from ill-conditioned matrices are held to what is true of them
//! (`MAY_REFUSE`, `LOOSE`): this port refuses a singular moment matrix with a typed error where
//! numpy returns digits or raises `LinAlgError`.

use raptorbt::math::linalg::Mat;
use raptorbt::stats::coint::*;
use raptorbt::stats::StatsError;
use serde_json::Value;

const EXTRACT: &str = include_str!("oracle/stats_coint_extract.json");

/// Reference answers that statsmodels produced from ill-conditioned moment matrices. A typed
/// refusal is accepted; so is an answer with the statistic's structure intact.
const MAY_REFUSE: [&str; 3] =
    ["collinear_near", "rank2_panel_3_series_noise_1e-09", "rank2_panel_5_series_noise_0.001"];
/// No panel is held to a looser Johansen bar than 1e-8: a panel whose moment matrices are too
/// ill-conditioned for it is refused (`coint::MAX_CONDITION`). The 5-series rank-2 panel at
/// noise 1e-3 (condition 6e7) is one: 60-digit eigenvalues sit 1.5e-8 from this port's and
/// 4.0e-8 from numpy's.
const LOOSE: [(&str, f64); 0] = [];
/// Intercepts the problem determines only to `eps * cond(X) * max|y|`, held to the 60-digit
/// value. `rank2_panel_3_series_noise_1e-09`: the intercept is -2.32140853814e-11 (mpmath, 80
/// digits, normal equations); statsmodels returns -2.32004971e-11 (5.9e-4 off; -2.32091013e-11
/// on another LAPACK build), this port -2.3215e-11.
/// cond(X) = 62.7 and max|y| = 25.96 give a bound of 2.2e-16 * 63 * 25.96 = 3.6e-13, three orders
/// above either error: the oracle's relative digits on a 2e-11 intercept are not meaningful.
const EXACT_INTERCEPT: [(&str, f64, f64); 1] =
    [("rank2_panel_3_series_noise_1e-09", -2.32140853814e-11, 3.6e-13)];
/// Panels numpy refuses that this port refuses too (a moment matrix singular to working
/// precision): a typed `degenerate_series`.
const REFUSED: [&str; 6] = [
    "collinear_exact",
    "duplicate_regressors",
    "constant_series",
    "rank2_panel_3_series_noise_1e-07",
    "rank2_panel_4_series_noise_1e-09",
    "rank2_panel_5_series_noise_1e-05",
];

fn hex_f64s(s: &str) -> Vec<f64> {
    s.as_bytes()
        .chunks(16)
        .map(|c| f64::from_bits(u64::from_str_radix(std::str::from_utf8(c).unwrap(), 16).unwrap()))
        .collect()
}

fn arr(v: &Value) -> Vec<f64> {
    hex_f64s(v["f8"].as_str().unwrap_or_else(|| panic!("not an f8 array: {v}")))
}

fn scalar(v: &Value) -> f64 {
    arr(v)[0]
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    a == b || (a - b).abs() <= tol * b.abs().max(1.0)
}

fn assert_close(what: &str, a: f64, b: f64, tol: f64) {
    assert!(close(a, b, tol), "{what}: {a:e} against {b:e} (tolerance {tol:e})");
}

/// Relative to the expected value, with a floor for results that are rounding noise around 0.
fn assert_rel(what: &str, a: f64, b: f64, tol: f64) {
    assert!(
        a == b || (a - b).abs() <= tol * b.abs() + 1e-15,
        "{what}: {a:e} against {b:e} (relative tolerance {tol:e})"
    );
}

fn assert_crit(what: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (a, b) in got.iter().zip(want) {
        assert_close(what, *a, *b, 1e-12);
    }
}

fn cases() -> Vec<Value> {
    let v: Value = serde_json::from_str(EXTRACT).unwrap();
    v["cases"].as_array().unwrap().clone()
}

fn named(name: &str) -> Value {
    cases().into_iter().find(|c| c["name"] == name).unwrap_or_else(|| panic!("no case {name}"))
}

/// The series of a cointegration case, in the order the reference received them.
fn columns(case: &Value) -> Vec<Vec<f64>> {
    let input = &case["input"]["args"][0];
    input["order"]
        .as_array()
        .unwrap()
        .iter()
        .map(|k| arr(&input["columns"][k.as_str().unwrap()]))
        .collect()
}

fn refs(cols: &[Vec<f64>]) -> Vec<&[f64]> {
    cols.iter().map(|c| c.as_slice()).collect()
}

fn matrix(cols: &[Vec<f64>]) -> Mat {
    let (n, q) = (cols[0].len(), cols.len());
    Mat::from_vec(n, q, (0..n * q).map(|p| cols[p % q][p / q]).collect()).unwrap()
}

fn loose(name: &str) -> Option<f64> {
    LOOSE.iter().find(|(n, _)| *n == name).map(|(_, t)| *t)
}

/// Signs each column so that its largest-magnitude entry is positive, as `johansen` does.
fn sign_normalised(flat: Vec<f64>, q: usize) -> Vec<Vec<f64>> {
    let mut cols = Vec::new();
    for j in 0..q {
        let mut col: Vec<f64> = (0..q).map(|i| flat[i * q + j]).collect();
        let big = (0..q).fold(0, |b, i| if col[i].abs() > col[b].abs() { i } else { b });
        if col[big] < 0.0 {
            col.iter_mut().for_each(|v| *v = -*v);
        }
        cols.push(col);
    }
    cols
}

/// Engle-Granger against `coint` and `OLS`.
#[test]
fn engle_granger_replays_statsmodels() {
    let mut replayed = 0;
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let lib = &case["lib"];
        if lib.is_null() {
            continue;
        }
        let cols = columns(&case);
        if cols.len() < 2 || REFUSED.contains(&name) {
            continue;
        }
        let eg_lib = &lib["engle_granger"];
        let parts = &lib["engle_granger_parts"];
        let got = match engle_granger(&cols[0], &refs(&cols[1..]), ALPHA) {
            Ok(g) => g,
            Err(e) => {
                assert!(
                    case["kind"] == "degenerate",
                    "{name}: engle_granger refused a {} case: {e}",
                    case["kind"]
                );
                assert!(matches!(e, StatsError::DegenerateSeries(_)), "{name}: {e:?}");
                continue;
            }
        };
        let tol = if case["kind"] == "degenerate" { 1e-5 } else { 1e-8 };
        // the pieces
        assert_crit(&format!("{name} critical"), &got.critical_values, &arr(&parts["crit_2010"]));
        if parts.get("collinear_branch").is_some() {
            assert_eq!(got.statistic, f64::NEG_INFINITY, "{name}");
            assert_eq!(got.adf_lags, None, "{name}");
            assert_eq!(got.p_value, if cols.len() <= 6 { Some(0.0) } else { None }, "{name}");
        } else {
            let adf = &parts["adf_on_residual"];
            assert_eq!(got.adf_lags.map(|l| l as u64), adf["lags"].as_u64(), "{name}: lags");
            assert_close(
                &format!("{name} statistic"),
                got.statistic,
                scalar(&adf["statistic"]),
                tol,
            );
        }
        // the numbers coint returns
        if eg_lib.get("error").is_none() {
            assert_close(
                &format!("{name} coint statistic"),
                got.statistic,
                scalar(&eg_lib["statistic"]),
                tol,
            );
            assert_close(
                &format!("{name} coint p"),
                got.p_value.unwrap(),
                scalar(&eg_lib["p_value"]),
                tol,
            );
            assert_crit(
                &format!("{name} coint critical"),
                &got.critical_values,
                &arr(&eg_lib["critical_values"]),
            );
        } else {
            // statsmodels' mackinnonp raises IndexError from 7 series: the port answers
            assert_eq!(eg_lib["error"]["type"], "IndexError", "{name}");
            assert!(cols.len() >= 7, "{name}");
            assert_eq!(got.p_value, None, "{name}");
            assert_eq!(got.cointegrated, Some(got.statistic < got.critical_values[1]), "{name}");
        }
        // the regression
        let params = arr(&lib["ols"]["params"]);
        match EXACT_INTERCEPT.iter().find(|(n, _, _)| *n == name) {
            Some((_, exact, bound)) => assert!(
                (got.intercept - exact).abs() <= *bound,
                "{name}: intercept {:e} against exact {exact:e}",
                got.intercept
            ),
            None => assert_rel(&format!("{name} intercept"), got.intercept, params[0], tol),
        }
        for (i, h) in got.hedge_ratios.iter().enumerate() {
            assert_rel(&format!("{name} hedge ratio {i}"), *h, params[i + 1], tol);
        }
        replayed += 1;
    }
    assert!(replayed >= 15, "only {replayed} cases replayed");
}

/// The reference's full-precision answer: spread statistics, half-life, decision.
#[test]
fn cointegration_matches_the_reference() {
    let mut checked = 0;
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let Some(ok) = case["reference"].get("ok").filter(|o| o.get("engle_granger").is_some())
        else {
            continue;
        };
        let cols = columns(&case);
        let got = match cointegration(&refs(&cols), ALPHA) {
            Ok(g) => g,
            Err(e) => {
                assert!(MAY_REFUSE.contains(&name), "{name}: refused: {e}");
                assert!(matches!(e, StatsError::DegenerateSeries(_)), "{name}: {e:?}");
                continue;
            }
        };
        let tol = loose(name).unwrap_or(1e-8);
        let eg = &got.engle_granger;
        let want = &ok["engle_granger"];
        assert_eq!(got.n as u64, ok["n"].as_u64().unwrap(), "{name}");
        if want["statistic"].is_null() {
            // a statistic of -inf: the series are (almost) one line
            assert!(eg.statistic.is_infinite() || !eg.statistic.is_finite(), "{name}");
        } else {
            assert_close(
                &format!("{name} statistic"),
                eg.statistic,
                scalar(&want["statistic"]),
                tol,
            );
        }
        assert_close(&format!("{name} p"), eg.p_value.unwrap(), scalar(&want["p_value"]), tol);
        assert_eq!(eg.cointegrated, want["cointegrated"].as_bool(), "{name}: decision");
        assert_rel(&format!("{name} intercept"), eg.intercept, scalar(&want["intercept"]), tol);
        for (i, k) in case["input"]["args"][0]["order"].as_array().unwrap()[1..].iter().enumerate()
        {
            assert_rel(
                &format!("{name} hedge {k}"),
                eg.hedge_ratios[i],
                scalar(&want["hedge_ratios"][k.as_str().unwrap()]),
                tol,
            );
        }
        // the spread's mean is rounding noise around 0: compared absolutely
        assert!(
            (eg.spread_mean - scalar(&want["spread_mean"])).abs() <= 1e-8 * eg.spread_std.max(1.0),
            "{name}: spread mean {:e} against {:e}",
            eg.spread_mean,
            scalar(&want["spread_mean"])
        );
        assert_rel(&format!("{name} spread std"), eg.spread_std, scalar(&want["spread_std"]), tol);
        if want["spread_last_z"].is_null() {
            continue;
        }
        assert_close(
            &format!("{name} last z"),
            eg.spread_last_z,
            scalar(&want["spread_last_z"]),
            tol.max(1e-7),
        );
        match (&want["half_life_periods"], eg.half_life_periods) {
            (Value::Null, None) => {}
            (v, Some(h)) => assert_close(&format!("{name} half-life"), h, scalar(v), tol.max(1e-7)),
            (v, None) => panic!("{name}: no half-life against {v}"),
        }
        // the Johansen block
        let jo = &ok["johansen"];
        let jtol = loose(name).unwrap_or(1e-8).max(if loose(name).is_some() { 1e-4 } else { 0.0 });
        assert_eq!(
            got.johansen.rank_at_5pct as u64,
            jo["rank_at_5pct"].as_u64().unwrap(),
            "{name}: rank"
        );
        for (a, b) in got.johansen.trace_statistics.iter().zip(arr(&jo["trace_statistics"])) {
            assert_close(&format!("{name} trace"), *a, b, jtol);
        }
        assert_eq!(got.johansen.trace_critical_5pct, arr(&jo["trace_critical_5pct"]), "{name}");
        if loose(name).is_none() {
            for (i, k) in case["input"]["args"][0]["order"].as_array().unwrap().iter().enumerate() {
                assert_close(
                    &format!("{name} vector {k}"),
                    got.johansen.cointegrating_vector[i],
                    scalar(&jo["cointegrating_vector"][k.as_str().unwrap()]),
                    1e-6,
                );
            }
        }
        checked += 1;
    }
    assert!(checked >= 9, "only {checked} cases checked");
}

/// Johansen: eigenvalues, vectors, statistics and tables, for every variant the oracle ran.
#[test]
#[allow(clippy::needless_range_loop)]
fn johansen_replays_statsmodels() {
    let mut replayed = 0;
    let mut refused = std::collections::BTreeSet::new();
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        if case["lib"].is_null() {
            continue;
        }
        let cols = columns(&case);
        let m = matrix(&cols);
        let q = cols.len();
        for v in case["lib"]["johansen"].as_array().unwrap() {
            if q < 2 {
                continue; // the reference refuses one series before Johansen
            }
            let (det, k) = (
                v["det_order"].as_i64().unwrap() as i32,
                v["k_ar_diff"].as_u64().unwrap() as usize,
            );
            let label = format!("{name} (det_order {det}, k_ar_diff {k})");
            let got = johansen(&m, det, k);
            // squared canonical correlations lie in [0, 1): statsmodels returns others only from
            // a moment matrix singular to working precision (which LAPACK build decides whether
            // it raises instead), and the port must refuse such a panel as it refuses an error
            let eig_invalid =
                v.get("eig").is_some_and(|e| arr(e).iter().any(|x| !(0.0..1.0).contains(x)));
            if v.get("error").is_some() || eig_invalid {
                assert!(
                    REFUSED.contains(&name) || (eig_invalid && MAY_REFUSE.contains(&name)),
                    "{label}: unexpected oracle error"
                );
                match got {
                    Err(StatsError::DegenerateSeries(msg)) => {
                        assert!(!msg.is_empty());
                        refused.insert(name.to_string());
                    }
                    other => panic!("{label}: expected a degenerate_series refusal, got {other:?}"),
                }
                continue;
            }
            let got = match got {
                Ok(j) => j,
                Err(e) => {
                    assert!(MAY_REFUSE.contains(&name), "{label}: refused: {e}");
                    assert!(matches!(e, StatsError::DegenerateSeries(_)), "{label}: {e:?}");
                    refused.insert(name.to_string());
                    continue;
                }
            };
            let tol = loose(name).unwrap_or(1e-8);
            for (a, b) in got.eig.iter().zip(arr(&v["eig"])) {
                assert_close(&format!("{label} eigenvalue"), *a, b, tol);
            }
            for (a, b) in got.lr1.iter().zip(arr(&v["lr1"])) {
                assert_close(
                    &format!("{label} trace"),
                    *a,
                    b,
                    tol.max(if loose(name).is_some() { 1e-4 } else { 0.0 }),
                );
            }
            for (a, b) in got.lr2.iter().zip(arr(&v["lr2"])) {
                assert_close(
                    &format!("{label} max eigenvalue"),
                    *a,
                    b,
                    tol.max(if loose(name).is_some() { 1e-4 } else { 0.0 }),
                );
            }
            // the tables: exact
            let (cvt, cvm) = (arr(&v["cvt"]), arr(&v["cvm"]));
            for i in 0..q {
                assert_eq!(
                    got.cvt[i].to_vec(),
                    cvt[i * 3..i * 3 + 3].to_vec(),
                    "{label}: trace critical"
                );
                assert_eq!(
                    got.cvm[i].to_vec(),
                    cvm[i * 3..i * 3 + 3].to_vec(),
                    "{label}: max-eig critical"
                );
            }
            if loose(name).is_none() {
                let want = sign_normalised(arr(&v["evec"]), q);
                for j in 0..q {
                    let scale = want[j].iter().fold(0.0f64, |m, x| m.max(x.abs()));
                    for i in 0..q {
                        let (a, b) = (got.evec[(i, j)], want[j][i]);
                        assert!(
                            (a - b).abs() <= 1e-6 * scale,
                            "{label}: eigenvector {j} entry {i}: {a:e} against {b:e}"
                        );
                    }
                }
            }
            assert_eq!(got.nobs, m.rows() - 1 - k, "{label}");
            replayed += 1;
        }
    }
    assert!(replayed >= 25, "only {replayed} fits replayed");
    // exactly the panels named above are refused, no others: a refusal cannot hollow the replay
    let expected: std::collections::BTreeSet<String> =
        REFUSED.iter().chain(&MAY_REFUSE).map(|n| n.to_string()).collect();
    assert_eq!(refused, expected);
}

/// The `johansen` block of `cointegration` carries statsmodels' `.eig`, `.lr2` and `.cvm[:, 1]`
/// (constant in the relation, one lagged difference) as the oracle recorded them.
#[test]
fn cointegration_summary_carries_the_eigenvalues_and_max_eigenvalue_statistics() {
    let mut checked = 0;
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        if case["lib"].is_null() {
            continue;
        }
        let cols = columns(&case);
        if cols.len() < 2 {
            continue;
        }
        let Some(v) = case["lib"]["johansen"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["det_order"].as_i64() == Some(0) && v["k_ar_diff"].as_u64() == Some(1))
        else {
            continue;
        };
        if v.get("error").is_some() {
            continue;
        }
        let Ok(got) = cointegration(&refs(&cols), ALPHA) else { continue };
        let jo = &got.johansen;
        let (eig, lr2) = (arr(&v["eig"]), arr(&v["lr2"]));
        assert_eq!(jo.eigenvalues.len(), eig.len(), "{name}");
        for (a, b) in jo.eigenvalues.iter().zip(&eig) {
            assert_close(&format!("{name} eigenvalue"), *a, *b, 1e-8);
        }
        for (a, b) in jo.max_eigenvalue_statistics.iter().zip(&lr2) {
            assert_close(&format!("{name} max eigenvalue statistic"), *a, *b, 1e-8);
        }
        let cvm = arr(&v["cvm"]);
        let want: Vec<f64> = (0..cols.len()).map(|i| cvm[i * 3 + 1]).collect();
        assert_eq!(jo.max_eigenvalue_critical_5pct, want, "{name}: 5% critical values");
        assert!(jo.eigenvalues.windows(2).all(|w| w[0] >= w[1]), "{name}: descending");
        checked += 1;
    }
    assert!(checked >= 9, "only {checked} cases checked");
}

/// The refusals of the reference and of the numerics, as typed errors.
#[test]
fn refusals_are_typed() {
    let go = |name: &str| {
        let cols = columns(&named(name));
        cointegration(&refs(&cols), ALPHA)
    };
    assert!(matches!(
        go("below_min_obs"),
        Err(StatsError::InsufficientObservations { found: 59, needed: 60, .. })
    ));
    assert!(matches!(go("single_series"), Err(StatsError::Invalid(_))));
    assert!(matches!(go("13_series"), Err(StatsError::Invalid(m)) if m.contains("at most 12")));
    assert!(matches!(go("constant_series"), Err(StatsError::DegenerateSeries(_))));
    // statsmodels lets numpy's LinAlgError out of these two
    for name in ["collinear_exact", "duplicate_regressors"] {
        let err = go(name).unwrap_err();
        assert!(matches!(err, StatsError::DegenerateSeries(_)), "{name}: {err:?}");
        assert_eq!(err.code(), "degenerate_series");
    }
    // collinear to rounding: the first step still answers, with the collinear branch's -inf
    let cols = columns(&named("collinear_exact"));
    let eg = engle_granger(&cols[0], &refs(&cols[1..]), ALPHA).unwrap();
    assert_eq!((eg.statistic, eg.p_value, eg.adf_lags), (f64::NEG_INFINITY, Some(0.0), None));
    assert_eq!(eg.cointegrated, Some(true));
    let cols = columns(&named("independent_walks"));
    let m = matrix(&cols);
    assert!(matches!(johansen(&m, 2, 1), Err(StatsError::Invalid(_))));
    assert!(matches!(johansen(&m, 0, 100), Err(StatsError::InsufficientObservations { .. })));
}

/// From 7 series statsmodels raises `IndexError`; the port answers with no p-value.
#[test]
fn seven_or_more_series_answer_without_a_p_value() {
    for (name, q) in [("7_series", 7), ("12_series", 12)] {
        let case = named(name);
        assert_eq!(case["reference"]["error"]["type"], "IndexError");
        let cols = columns(&case);
        assert_eq!(cols.len(), q);
        let got = cointegration(&refs(&cols), ALPHA).unwrap_or_else(|e| panic!("{name}: {e}"));
        let eg = &got.engle_granger;
        assert_eq!(eg.p_value, None);
        assert_eq!(eg.cointegrated, Some(eg.statistic < eg.critical_values[1]));
        assert_crit(
            name,
            &eg.critical_values,
            &arr(&case["lib"]["engle_granger_parts"]["crit_2010"]),
        );
        assert_eq!(got.johansen.trace_statistics.len(), q);
        // another level than the three the critical values are tabulated at: no decision
        let at_3pct = engle_granger(&cols[0], &refs(&cols[1..]), 0.03).unwrap();
        assert_eq!(at_3pct.cointegrated, None);
        let at_1pct = engle_granger(&cols[0], &refs(&cols[1..]), 0.01).unwrap();
        assert_eq!(at_1pct.cointegrated, Some(at_1pct.statistic < at_1pct.critical_values[0]));
    }
}

/// Half-life of the spread's AR(1).
#[test]
fn half_life_replays_the_reference_and_hand_values() {
    let exact = named("half_life_exact_decay");
    let spread = arr(&exact["input"]["args"][0]);
    // s_t = 0.9 s_{t-1} exactly, so phi = 0.9 and the half-life is ln 2 / -ln 0.9
    let want = scalar(&exact["known"]["half_life"]);
    assert_close("half-life", half_life(&spread).unwrap(), want, 1e-9);
    // s_t = 0.5^t: the change is -0.5 times the lagged level, phi = 0.5, one period to halve
    let halving: Vec<f64> = (0..40).map(|t| 0.5f64.powi(t)).collect();
    assert_close("halving", half_life(&halving).unwrap(), 1.0, 1e-9);
    // a spread that trends has no half-life, nor has a constant one or a short one
    let flat = named("half_life_trending_spread");
    assert_eq!(half_life(&arr(&flat["input"]["args"][0])), None);
    assert_eq!(half_life(&[1.0; 20]), None);
    assert_eq!(half_life(&[1.0, 2.0]), None);
}

/// With the noise projected off the regressors the hedge ratio is exact, to rounding.
#[test]
fn known_hedge_ratio_is_recovered() {
    let case = named("known_hedge_ratio");
    let cols = columns(&case);
    let eg = engle_granger(&cols[0], &refs(&cols[1..]), ALPHA).unwrap();
    assert_close("intercept", eg.intercept, scalar(&case["known"]["intercept"]), 1e-9);
    assert_close("hedge ratio", eg.hedge_ratios[0], scalar(&case["known"]["hedge_ratio"]), 1e-9);
}

/// Each eigenvector's largest entry is positive, the eigenvalues descend inside (0, 1), and the
/// trace statistic is the tail sum the definition gives.
#[test]
fn johansen_invariants() {
    let cols = columns(&named("three_series"));
    let m = matrix(&cols);
    let j = johansen(&m, 0, 1).unwrap();
    assert!(j.eig.windows(2).all(|w| w[0] >= w[1]));
    assert!(j.eig.iter().all(|a| *a > 0.0 && *a < 1.0));
    for c in 0..3 {
        let col = j.evec.col(c);
        let big = col.iter().fold(0.0f64, |m, x| if x.abs() > m.abs() { *x } else { m });
        assert!(big > 0.0, "column {c}: {col:?}");
    }
    let t = j.nobs as f64;
    for i in 0..3 {
        let by_hand: f64 = -t * j.eig[i..].iter().map(|a| (1.0 - a).ln()).sum::<f64>();
        assert_close("trace", j.lr1[i], by_hand, 1e-12);
        assert_close("max eigenvalue", j.lr2[i], -t * (1.0 - j.eig[i]).ln(), 1e-12);
    }
    // the trace of the first hypothesis is the sum of all the max-eigenvalue statistics
    assert_close("sum", j.lr1[0], j.lr2.iter().sum(), 1e-10);
}

/// Scaling a series changes no eigenvalue and rescales its entry of every eigenvector.
#[test]
fn johansen_is_invariant_to_units() {
    let cols = columns(&named("cointegrated_pair"));
    let base = johansen(&matrix(&cols), 0, 1).unwrap();
    let mut scaled = cols.clone();
    scaled[1] = scaled[1].iter().map(|v| v * 1000.0).collect();
    let got = johansen(&matrix(&scaled), 0, 1).unwrap();
    for (a, b) in got.eig.iter().zip(&base.eig) {
        assert_close("eigenvalue", *a, *b, 1e-9);
    }
    for j in 0..2 {
        // the second series is a thousand times larger, so its coefficient is a thousand times
        // smaller; the sign rule picks the largest entry, which may swap, so compare up to sign
        let (b0, b1) = (base.evec[(0, j)], base.evec[(1, j)]);
        let (g0, g1) = (got.evec[(0, j)], got.evec[(1, j)] * 1000.0);
        let sign = if (g0 * b0 + g1 * b1) < 0.0 { -1.0 } else { 1.0 };
        let scale = b0.abs().max(b1.abs());
        assert!(
            (g0 - sign * b0).abs() <= 1e-6 * scale && (g1 - sign * b1).abs() <= 1e-6 * scale,
            "column {j}"
        );
    }
}

fn walks(q: usize, n: usize) -> Vec<Vec<f64>> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    (0..q)
        .map(|_| {
            let mut level = 100.0;
            (0..n)
                .map(|_| {
                    state =
                        state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    level += ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 3.4;
                    level
                })
                .collect()
        })
        .collect()
}

/// Reinsel and Ahn's factor is `(T - q k) / T` with `T = n - 2` and `k = 2`; the scaled statistics
/// are the raw ones times it; the raw ones and `rank_at_5pct` are not touched; a smaller statistic
/// cannot give a larger rank.
#[test]
fn the_scaled_trace_statistics_and_the_small_sample_flag() {
    for (q, n, small) in [(12usize, 60usize, true), (6, 60, true), (2, 250, false), (4, 120, false)]
    {
        let cols = walks(q, n);
        let refs: Vec<&[f64]> = cols.iter().map(|c| c.as_slice()).collect();
        let r = cointegration(&refs, 0.05).unwrap().johansen;
        let t = (n - 2) as f64;
        let f = (t - (2 * q) as f64) / t;
        assert_eq!(r.effective_observations, n - 2);
        assert_eq!(r.scale_factor, f, "q={q} n={n}");
        assert_eq!(r.small_sample, small, "q={q} n={n}");
        for (s, raw) in r.trace_statistics_scaled.iter().zip(&r.trace_statistics) {
            assert_eq!(*s, raw * f);
        }
        assert!(r.rank <= r.rank_at_5pct, "q={q} n={n}");
    }
}
