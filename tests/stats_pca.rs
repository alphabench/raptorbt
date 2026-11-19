//! Principal components and the factor regression against numpy's SVD and least squares at
//! 1e-8 (`oracle/stats_pca_extract.json`). A component is compared as the reference gives it when
//! its raw loading sum is clearly nonzero, and up to sign when that sum is within 1e-6 of zero:
//! there the sign is a convention (the reference's and this crate's are stated in
//! `stats_reference.py` and `pca.rs`), not something the data decide.

use raptorbt::stats::pca::{pca_factors, Pca};
use raptorbt::stats::StatsError;
use serde_json::Value;

const TOL: f64 = 1e-8;

fn extract() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/oracle/stats_pca_extract.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("extract")).expect("json")
}

fn bits(hex: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(hex, 16).expect("hex digits"))
}

fn f8(v: &Value) -> Vec<f64> {
    let hex = v["f8"].as_str().expect("f8 array");
    (0..hex.len() / 16).map(|i| bits(&hex[i * 16..i * 16 + 16])).collect()
}

fn num(v: &Value) -> f64 {
    v.get("f8").map_or_else(|| v.as_f64().expect("number"), |h| bits(h.as_str().expect("hex")))
}

/// Within TOL of the value, scaled by at least 1 (loadings and statistics are of that size, and
/// a null component's ratio is a rounding error from zero).
fn close(what: &str, got: f64, want: f64) {
    assert!((got - want).abs() <= TOL * want.abs().max(1.0), "{what}: got {got:e}, want {want:e}");
}

fn columns(v: &Value) -> Vec<Vec<f64>> {
    v["order"].as_array().unwrap().iter().map(|n| f8(&v["columns"][n.as_str().unwrap()])).collect()
}

/// +1 when the component is compared as given, otherwise the sign that matches the reference's
/// vector.
fn sign_of(got: &[f64], want: &[f64], row_sum: f64) -> f64 {
    if row_sum.abs() > 1e-6 {
        return 1.0;
    }
    let dot: f64 = got.iter().zip(want).map(|(a, b)| a * b).sum();
    if dot < 0.0 {
        -1.0
    } else {
        1.0
    }
}

fn same(name: &str, got: &Pca, c: &Value) {
    let want = &c["expect"];
    let names: Vec<&str> = c["input"]["args"][0]["order"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_str().unwrap())
        .collect();
    assert_eq!(got.n as u64, want["n"].as_u64().unwrap(), "{name}");
    assert_eq!(got.components as u64, want["components"].as_u64().unwrap(), "{name}");
    for (j, w) in f8(&want["explained_variance_ratio"]).iter().enumerate() {
        close(&format!("{name} ratio[{j}]"), got.explained_variance_ratio[j], *w);
    }
    close(name, got.cumulative_explained, num(&want["cumulative_explained"]));
    let sums = f8(&c["lib"]["row_sums_raw"]);
    let mut signs = Vec::new();
    for j in 0..got.components {
        let reference = &want["loadings"][format!("pc{}", j + 1)];
        let w: Vec<f64> = names.iter().map(|n| num(&reference[*n])).collect();
        let sign = sign_of(&got.loadings[j], &w, sums[j]);
        for (i, wi) in w.iter().enumerate() {
            close(
                &format!("{name} loading pc{} {}", j + 1, names[i]),
                sign * got.loadings[j][i],
                *wi,
            );
        }
        signs.push(sign);
    }
    if signs.iter().all(|s| *s == 1.0) {
        assert_eq!(
            got.first_component_is_common_factor,
            want["first_component_is_common_factor"].as_bool().unwrap(),
            "{name}"
        );
    }
    match (&got.target, want.get("target")) {
        (None, None) => {}
        (Some(t), Some(w)) => {
            // a rank-deficient panel: `known` holds the OLS t-statistics on the components with
            // variance, recomputed from the inputs (inverting the nearly singular matrix instead
            // is wrong for every component)
            let exact = c.get("known").filter(|k| k.get("t_abs").is_some());
            close(&format!("{name} r2"), t.r_squared, num(&w["r_squared"]));
            close(&format!("{name} alpha"), t.alpha_per_period, num(&w["alpha_per_period"]));
            match exact {
                Some(k) => {
                    close(&format!("{name} alpha_t"), t.alpha_t.abs(), f8(&k["alpha_t_abs"])[0])
                }
                None => close(&format!("{name} alpha_t"), t.alpha_t, num(&w["alpha_t"])),
            }
            for (j, e) in t.exposures_per_unit_score.iter().enumerate() {
                let reference = &w["exposures_per_unit_score"][format!("pc{}", j + 1)];
                close(
                    &format!("{name} beta pc{}", j + 1),
                    signs[j] * e.beta,
                    num(&reference["beta"]),
                );
                match exact {
                    Some(k) => {
                        close(&format!("{name} t pc{}", j + 1), e.t.abs(), f8(&k["t_abs"])[j])
                    }
                    None => close(
                        &format!("{name} t pc{}", j + 1),
                        signs[j] * e.t,
                        num(&reference["t"]),
                    ),
                }
            }
        }
        (g, w) => panic!("{name}: target {g:?} against {w:?}"),
    }
}

#[test]
fn replays_the_fixture() {
    let doc = extract();
    let mut ok = 0;
    for c in doc["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let cols = columns(&c["input"]["args"][0]);
        let kw = &c["input"]["kwargs"];
        let components = kw.get("components").map_or(3, |v| v.as_u64().unwrap() as usize);
        // a lone array is {"f8"}; the harness stored a kwargs dict made only of arrays (the
        // wrong-length target) as a `Columns`
        let target = kw.get("target").or_else(|| kw["columns"].get("target")).map(f8);
        let got = pca_factors(&cols, components, target.as_deref());
        if let Some(msg) = c.get("error") {
            let err = got.expect_err(name);
            let want_code = match msg.as_str().unwrap() {
                m if m.contains("this statistic needs") => "insufficient_observations",
                "a series is constant" => "degenerate_series",
                _ => "invalid_input",
            };
            assert_eq!(err.code(), want_code, "{name}: {err}");
            continue;
        }
        same(name, &got.unwrap_or_else(|e| panic!("{name}: {e}")), c);
        ok += 1;
    }
    assert!(ok >= 9, "only {ok} value cases replayed");
}

#[test]
fn known_two_series_has_the_closed_form() {
    let doc = extract();
    let c =
        doc["cases"].as_array().unwrap().iter().find(|c| c["name"] == "known_two_series").unwrap();
    let got = pca_factors(&columns(&c["input"]["args"][0]), 2, None).unwrap();
    let known = &c["known"];
    // eigenvalues of a 2 x 2 correlation matrix are 1 +- rho; the loading is 1 / sqrt(2)
    close("first ratio", got.explained_variance_ratio[0], num(&known["first_ratio"]));
    close("second ratio", got.explained_variance_ratio[1], num(&known["second_ratio"]));
    close("loading", got.loadings[0][0], num(&known["loading"]));
}

/// Two opposite-sign series: the first component's raw sum is exactly zero, and the sign no longer
/// depends on the SVD. The loading of largest magnitude (the first, within SIGN_TOL) is positive.
#[test]
fn a_zero_loading_sum_gets_a_deterministic_sign() {
    let doc = extract();
    let c = doc["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "anticorrelated_pair")
        .unwrap();
    let cols = columns(&c["input"]["args"][0]);
    let got = pca_factors(&cols, 2, None).unwrap();
    assert!(got.loadings[0][0] > 0.0 && got.loadings[0][1] < 0.0, "{:?}", got.loadings[0]);
    assert!(!got.first_component_is_common_factor);
    // the same answer when the series are given in the other order, mirrored
    let flipped = vec![cols[1].clone(), cols[0].clone()];
    let other = pca_factors(&flipped, 2, None).unwrap();
    assert!(other.loadings[0][0] > 0.0 && other.loadings[0][1] < 0.0, "{:?}", other.loadings[0]);
}

#[test]
fn refuses_what_the_reference_refuses() {
    // a pattern of period 29 shifted 40 ways: rank 28, with repeated columns
    let make = |k: usize| -> Vec<Vec<f64>> {
        (0..k)
            .map(|j| (0..80).map(|t| (((t * 7 + j * 13) % 29) as f64 - 14.0) / 100.0).collect())
            .collect()
    };
    assert!(matches!(pca_factors(&make(41), 3, None), Err(StatsError::Invalid(_))));
    assert!(pca_factors(&make(40), 3, None).is_ok());
    let mut constant = make(3);
    constant[1] = vec![0.5; 80];
    assert_eq!(pca_factors(&constant, 3, None).unwrap_err().code(), "degenerate_series");
    // a constant target has no variance to explain
    let flat = vec![0.1; 80];
    assert_eq!(pca_factors(&make(3), 3, Some(&flat)).unwrap_err().code(), "degenerate_series");
}
