//! Newey-West t-statistic against statsmodels (`OLS(r, 1).fit(cov_type="HAC")`) at 1e-10:
//! the replay of `oracle/stats_hac_extract.json`, cut from the fixture by
//! `oracle/extract_stats.py`. The HAC p-value is normal, the naive one is t.

use raptorbt::stats::hac::hac_tstat;
use raptorbt::stats::StatsError;
use serde_json::Value;

const TOL: f64 = 1e-10;

fn extract() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/oracle/stats_hac_extract.json");
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
    rel_to(what, got, want, TOL);
}

fn rel_to(what: &str, got: f64, want: f64, tol: f64) {
    let err = (got - want).abs() / want.abs().max(1e-300);
    assert!(err <= tol, "{what}: got {got:e}, want {want:e}, rel {err:e}");
}

#[test]
fn replays_the_fixture() {
    let doc = extract();
    let mut ok = 0;
    for c in doc["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let returns = f8(&c["input"]["args"][0]);
        let lags = c["input"]["kwargs"].get("lags").map(|l| l.as_i64().unwrap());
        let got = hac_tstat(&returns, lags);
        if let Some(msg) = c.get("error") {
            let err = got.expect_err(name);
            assert_eq!(err.to_string(), msg.as_str().unwrap(), "{name}");
            continue;
        }
        let got = got.unwrap_or_else(|e| panic!("{name}: {e}"));
        let want = &c["expect"];
        let lib = &c["lib"];
        assert_eq!(got.n as u64, want["n"].as_u64().unwrap(), "{name}");
        assert_eq!(got.lags as u64, want["lags"].as_u64().unwrap(), "{name}");
        assert_eq!(got.lags as u64, lib["lags_used"].as_u64().unwrap(), "{name}");
        // a mean of 1 with noise of 1e-9: statsmodels' own residuals carry the rounding error of
        // its mean, which is 1e-8 of the standard error (exact arithmetic, in `known`, is the
        // judge there); everywhere else the oracle is the bar
        let exact = c.get("known").and_then(|k| k.get("hac_se_exact"));
        let tol = if exact.is_some() { 1e-6 } else { TOL };
        rel(&format!("{name} mean"), got.mean, num(&lib["mean"]));
        rel_to(&format!("{name} hac_std_error"), got.hac_std_error, num(&lib["hac_bse"]), tol);
        rel_to(&format!("{name} hac_t"), got.hac_t, num(&lib["hac_t"]), tol);
        rel_to(&format!("{name} hac_p"), got.hac_p_value, num(&lib["hac_p_normal"]), tol);
        rel_to(&format!("{name} naive_t"), got.naive_t, num(&lib["naive_t"]), tol);
        rel_to(&format!("{name} naive_p"), got.naive_p_value, num(&lib["naive_p_t"]), tol);
        rel_to(
            &format!("{name} sharpe"),
            got.per_period_sharpe,
            num(&want["per_period_sharpe"]),
            tol,
        );
        rel_to(
            &format!("{name} lag1"),
            got.lag1_autocorrelation,
            num(&want["lag1_autocorrelation"]),
            tol,
        );
        if let Some(e) = exact {
            rel_to(&format!("{name} exact hac_se"), got.hac_std_error, num(e), 1e-12);
        }
        if let Some(known) = c.get("known").filter(|k| k.get("mean").is_some()) {
            rel(&format!("{name} known mean"), got.mean, num(&known["mean"]));
            rel(&format!("{name} known hac_se"), got.hac_std_error, num(&known["hac_se"]));
            rel(&format!("{name} known naive_se"), got.mean / got.naive_t, num(&known["naive_se"]));
        }
        ok += 1;
    }
    assert!(ok >= 9, "only {ok} value cases replayed");
}

#[test]
fn refusals_carry_their_code() {
    let short = vec![0.01; 29];
    assert_eq!(hac_tstat(&short, None).unwrap_err().code(), "insufficient_observations");
    assert_eq!(hac_tstat(&[0.01; 60], None).unwrap_err().code(), "degenerate_series");
    let noisy: Vec<f64> = (0..60).map(|i| ((i * 7) % 11) as f64 / 100.0).collect();
    assert!(matches!(hac_tstat(&noisy, Some(30)), Err(StatsError::Invalid(_))));
    let mut bad = noisy.clone();
    bad[3] = f64::NAN;
    assert!(matches!(hac_tstat(&bad, None), Err(StatsError::NonFinite { .. })));
}
