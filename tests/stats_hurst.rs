//! The Hurst exponent against the reference at 1e-10 with the scales exact (`oracle/stats_hurst_extract
//! .json`). The scales are `floor(logspace(...))`, so they are also compared with numpy's for
//! every length in `scale_sweep`.

use raptorbt::stats::hurst::{expected_rs, hurst, scales};
use serde_json::Value;

const TOL: f64 = 1e-10;

fn extract() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/oracle/stats_hurst_extract.json");
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

fn rel(what: &str, got: f64, want: f64) {
    let err = (got - want).abs() / want.abs().max(1e-300);
    assert!(err <= TOL, "{what}: got {got:e}, want {want:e}, rel {err:e}");
}

fn ints(v: &Value) -> Vec<usize> {
    if let Some(hex) = v.get("i4") {
        let hex = hex.as_str().unwrap();
        return (0..hex.len() / 8)
            .map(|i| usize::from_str_radix(&hex[i * 8..i * 8 + 8], 16).unwrap())
            .collect();
    }
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect()
}

#[test]
fn replays_the_fixture() {
    let doc = extract();
    let mut ok = 0;
    for c in doc["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let got = hurst(&f8(&c["input"]["args"][0]));
        if let Some(msg) = c.get("error") {
            assert_eq!(got.expect_err(name).to_string(), msg.as_str().unwrap(), "{name}");
            continue;
        }
        let got = got.unwrap_or_else(|e| panic!("{name}: {e}"));
        let want = &c["expect"];
        assert_eq!(got.n as u64, want["n"].as_u64().unwrap(), "{name}");
        let (rs, dfa) = (&want["rescaled_range"], &want["dfa"]);
        assert_eq!(got.rescaled_range.scales, ints(&rs["scales"]), "{name} R/S scales");
        assert_eq!(got.dfa.scales, ints(&dfa["scales"]), "{name} DFA scales");
        rel(&format!("{name} R/S hurst"), got.rescaled_range.hurst, num(&rs["hurst"]));
        rel(&format!("{name} R/S slope"), got.rescaled_range.raw_slope, num(&rs["raw_slope"]));
        rel(&format!("{name} R/S se"), got.rescaled_range.std_error, num(&rs["std_error"]));
        rel(&format!("{name} DFA hurst"), got.dfa.hurst, num(&dfa["hurst"]));
        rel(&format!("{name} DFA se"), got.dfa.std_error, num(&dfa["std_error"]));
        ok += 1;
    }
    assert!(ok >= 10, "only {ok} value cases replayed");
}

#[test]
fn expected_rs_matches_scipy_gamma_form_and_peters() {
    let doc = extract();
    let t = &doc["tables"];
    let (ns, want) = (ints(&t["expected_rs_n"]), f8(&t["expected_rs"]));
    assert!(ns.contains(&2) && ns.iter().any(|n| *n > 340), "the table spans the Peters switch");
    for (n, w) in ns.iter().zip(want) {
        rel(&format!("E[R/S] at {n}"), expected_rs(*n).unwrap(), w);
    }
}

#[test]
fn scales_are_numpys_floor_of_logspace() {
    let doc = extract();
    let table = doc["tables"]["scales"].as_object().unwrap();
    assert!(table.len() > 40);
    for (n, v) in table {
        assert_eq!(scales(n.parse().unwrap()), ints(v), "scales({n})");
    }
    let sweep = doc["scale_sweep"].as_object().unwrap();
    assert!(sweep.len() > 1500);
    for (n, v) in sweep {
        assert_eq!(scales(n.parse().unwrap()), ints(v), "scales({n})");
    }
}
