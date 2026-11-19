//! The bar audit against the reference's findings (`oracle/stats_quality_extract.json`): counts,
//! times and gap days exact, `return_pct` and `close_ratio` within 1e-12 (an `exp` of a difference
//! of logs, which two libms may round one ulp apart). A series of flat, zero-volume bars is not
//! clean (`all_flat`), and too few bars is a typed refusal.

use raptorbt::stats::quality::{audit_bars, Audit, AuditRules, Bars, Gap};
use serde_json::Value;

fn extract() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/oracle/stats_quality_extract.json");
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
    assert!(err <= 1e-12, "{what}: got {got:e}, want {want:e}, rel {err:e}");
}

fn run(c: &Value) -> Result<Audit, raptorbt::stats::StatsError> {
    let a = c["input"]["args"].as_array().unwrap();
    let cols: Vec<Vec<f64>> = a.iter().map(f8).collect();
    let bars = Bars {
        t: &cols[0],
        open: &cols[1],
        high: &cols[2],
        low: &cols[3],
        close: &cols[4],
        volume: &cols[5],
    };
    let interval = c["input"]["kwargs"]["interval_seconds"].as_u64().unwrap();
    audit_bars(&bars, interval, &AuditRules::default())
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect()
}

fn same(name: &str, got: &Audit, w: &Value) {
    assert_eq!(got.bars as u64, w["bars"].as_u64().unwrap(), "{name}");
    assert_eq!(got.first, w["first"].as_str().unwrap(), "{name}");
    assert_eq!(got.last, w["last"].as_str().unwrap(), "{name}");
    let (f, wf) = (&got.findings, &w["findings"]);
    assert_eq!(
        f.duplicate_timestamps as u64,
        wf["duplicate_timestamps"].as_u64().unwrap(),
        "{name}"
    );
    assert_eq!(f.out_of_order as u64, wf["out_of_order"].as_u64().unwrap(), "{name}");
    assert_eq!(
        f.impossible_ohlc.count as u64,
        wf["impossible_ohlc"]["count"].as_u64().unwrap(),
        "{name}"
    );
    assert_eq!(f.impossible_ohlc.first, strings(&wf["impossible_ohlc"]["first"]), "{name}");
    assert_eq!(f.zero_volume_bars as u64, wf["zero_volume_bars"].as_u64().unwrap(), "{name}");
    assert_eq!(f.flat_bars as u64, wf["flat_bars"].as_u64().unwrap(), "{name}");
    assert_eq!(
        f.flat_zero_volume_bars as u64,
        wf["flat_zero_volume_bars"].as_u64().unwrap(),
        "{name}"
    );
    assert_eq!(f.robust_z_over, num(&wf["return_outliers"]["robust_z_over"]), "{name}");

    let worst = wf["return_outliers"]["worst"].as_array().unwrap();
    assert_eq!(f.return_outliers.len(), worst.len(), "{name}: outliers");
    for (g, w) in f.return_outliers.iter().zip(worst) {
        assert_eq!(g.at, w["at"].as_str().unwrap(), "{name}");
        rel(&format!("{name} return_pct"), g.return_pct, num(&w["return_pct"]));
    }
    let splits = wf["possible_unadjusted_splits"].as_array().unwrap();
    assert_eq!(f.possible_unadjusted_splits.len(), splits.len(), "{name}: splits");
    for (g, w) in f.possible_unadjusted_splits.iter().zip(splits) {
        assert_eq!(g.at, w["at"].as_str().unwrap(), "{name}");
        rel(&format!("{name} close_ratio"), g.close_ratio, num(&w["close_ratio"]));
    }
    let gaps = wf["gaps"].as_array().unwrap();
    assert_eq!(f.gaps.len(), gaps.len(), "{name}: gaps");
    for (g, w) in f.gaps.iter().zip(gaps) {
        match g {
            Gap::Daily { from, to, days } => {
                assert_eq!(
                    (from.as_str(), to.as_str()),
                    (w["from"].as_str().unwrap(), w["to"].as_str().unwrap())
                );
                assert_eq!(*days, num(&w["days"]), "{name}");
            }
            Gap::Intraday { from, to, missing_bars } => {
                assert_eq!(
                    (from.as_str(), to.as_str()),
                    (w["from"].as_str().unwrap(), w["to"].as_str().unwrap())
                );
                assert_eq!(*missing_bars as u64, w["missing_bars"].as_u64().unwrap(), "{name}");
            }
        }
    }
}

#[test]
fn replays_the_fixture() {
    let doc = extract();
    let mut ok = 0;
    for c in doc["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let got = run(c);
        if c.get("error").is_some() {
            assert_eq!(got.expect_err(name).code(), "insufficient_observations", "{name}");
            continue;
        }
        let got = got.unwrap_or_else(|e| panic!("{name}: {e}"));
        let want = &c["expect"];
        same(name, &got, want);
        assert_eq!(got.clean, want["clean"].as_bool().unwrap(), "{name}");
        if name == "all_flat" {
            assert!(!got.clean, "{name}");
            assert_eq!(got.findings.flat_zero_volume_bars, got.bars, "{name}");
        }
        if let Some(known) = c.get("known") {
            let f = &got.findings;
            assert_eq!(
                f.duplicate_timestamps as u64,
                known["duplicate_timestamps"].as_u64().unwrap()
            );
            assert_eq!(f.flat_bars as u64, known["flat_bars"].as_u64().unwrap());
            assert_eq!(f.impossible_ohlc.count as u64, known["impossible_ohlc"].as_u64().unwrap());
            assert_eq!(f.out_of_order as u64, known["out_of_order"].as_u64().unwrap());
            assert_eq!(f.zero_volume_bars as u64, known["zero_volume_bars"].as_u64().unwrap());
        }
        ok += 1;
    }
    assert!(ok >= 12, "only {ok} value cases replayed");
}

/// The thresholds are parameters: tightening one changes the finding it governs.
#[test]
fn thresholds_are_parameters() {
    let doc = extract();
    let c = doc["cases"].as_array().unwrap().iter().find(|c| c["name"] == "daily_gap").unwrap();
    let a = c["input"]["args"].as_array().unwrap();
    let cols: Vec<Vec<f64>> = a.iter().map(f8).collect();
    let bars = Bars {
        t: &cols[0],
        open: &cols[1],
        high: &cols[2],
        low: &cols[3],
        close: &cols[4],
        volume: &cols[5],
    };
    let strict = audit_bars(&bars, 86_400, &AuditRules::default()).unwrap();
    assert!(!strict.findings.gaps.is_empty() && !strict.clean);
    let lax = AuditRules { daily_gap_days: 30.0, ..AuditRules::default() };
    let loose = audit_bars(&bars, 86_400, &lax).unwrap();
    assert!(loose.findings.gaps.is_empty());
}
