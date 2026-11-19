//! The probability of backtest overfitting against the reference's numbers (1e-12), replaying
//! `oracle/stats_pbo_extract.json`, plus the closed-form cases and the promise that the answer
//! does not depend on the number of threads.

use raptorbt::stats::pbo::{pbo_cscv, Pbo};
use raptorbt::stats::StatsError;
use serde_json::Value;

const TOL: f64 = 1e-12;

fn extract() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/oracle/stats_pbo_extract.json");
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

/// The columns of a `Columns` input in the order the reference was given them.
fn columns(v: &Value) -> Vec<Vec<f64>> {
    v["order"].as_array().unwrap().iter().map(|n| f8(&v["columns"][n.as_str().unwrap()])).collect()
}

fn same(name: &str, got: &Pbo, want: &Value) {
    assert_eq!(got.n_periods as u64, want["n_periods"].as_u64().unwrap(), "{name}");
    assert_eq!(got.n_strategies as u64, want["n_strategies"].as_u64().unwrap(), "{name}");
    assert_eq!(got.splits as u64, want["splits"].as_u64().unwrap(), "{name}");
    assert_eq!(got.combinations as u64, want["combinations"].as_u64().unwrap(), "{name}");
    rel(&format!("{name} pbo"), got.pbo, num(&want["pbo"]));
    rel(&format!("{name} median_logit"), got.median_logit, num(&want["median_logit"]));
    rel(
        &format!("{name} in-sample"),
        got.mean_in_sample_sharpe_of_best,
        num(&want["mean_in_sample_sharpe_of_best"]),
    );
    rel(
        &format!("{name} out-of-sample"),
        got.mean_out_of_sample_sharpe_of_best,
        num(&want["mean_out_of_sample_sharpe_of_best"]),
    );
    rel(
        &format!("{name} loss"),
        got.prob_out_of_sample_loss_of_best,
        num(&want["prob_out_of_sample_loss_of_best"]),
    );
}

#[test]
fn replays_the_fixture() {
    let doc = extract();
    let mut ok = 0;
    for c in doc["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let cols = columns(&c["input"]["args"][0]);
        let splits =
            c["input"]["kwargs"].get("splits").map_or(16, |s| s.as_u64().unwrap() as usize);
        let got = pbo_cscv(&cols, splits);
        if let Some(msg) = c.get("error") {
            let err = got.expect_err(name);
            let want_code = if msg.as_str().unwrap().contains("this statistic needs") {
                "insufficient_observations"
            } else {
                "invalid_input"
            };
            assert_eq!(err.code(), want_code, "{name}: {err}");
            continue;
        }
        let got = got.unwrap_or_else(|e| panic!("{name}: {e}"));
        same(name, &got, &c["expect"]);
        if let Some(known) = c.get("known") {
            rel(&format!("{name} known pbo"), got.pbo, num(&known["pbo"]));
            rel(&format!("{name} known logit"), got.median_logit, num(&known["median_logit"]));
            if let Some(loss) = known.get("prob_loss") {
                rel(&format!("{name} known loss"), got.prob_out_of_sample_loss_of_best, num(loss));
            }
        }
        ok += 1;
    }
    assert!(ok >= 10, "only {ok} value cases replayed");
}

#[test]
fn drops_the_oldest_rows_to_fit_the_blocks() {
    let doc = extract();
    let c = doc["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "rows_not_multiple_of_splits")
        .expect("case");
    let cols = columns(&c["input"]["args"][0]);
    let got = pbo_cscv(&cols, 16).unwrap();
    assert_eq!(got.dropped_rows + got.n_periods, cols[0].len());
    assert_eq!(got.dropped_rows, 10, "the 10 oldest rows");
}

fn synthetic(k: usize, rows: usize) -> Vec<Vec<f64>> {
    (0..k)
        .map(|j| {
            (0..rows)
                .map(|t| (((t * 31 + j * 17 + (t * j) % 13) % 97) as f64 - 48.0) / 1000.0)
                .collect()
        })
        .collect()
}

#[test]
fn caps_the_number_of_strategies() {
    assert!(pbo_cscv(&synthetic(200, 64), 16).is_ok());
    assert!(matches!(pbo_cscv(&synthetic(201, 64), 16), Err(StatsError::Invalid(_))));
}

#[test]
fn the_answer_does_not_depend_on_the_thread_count() {
    let cols = synthetic(40, 320);
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| pbo_cscv(&cols, 16).unwrap())
    };
    let one = run(1);
    assert_eq!(one, run(3));
    assert_eq!(one, run(8));
}
