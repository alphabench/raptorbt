//! The impact and spread estimates against the reference at 1e-12 (`oracle/stats_impact_extract.json`),
//! and the closed forms: flat bars give a Corwin-Schultz spread of exactly 0, alternating prices
//! a Roll spread from a covariance computed in exact rational arithmetic.

use raptorbt::stats::impact::{corwin_schultz, impact_estimate, roll, SpreadModel};
use serde_json::Value;

const TOL: f64 = 1e-12;

fn extract() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/oracle/stats_impact_extract.json");
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

#[test]
fn replays_the_fixture() {
    let doc = extract();
    let mut ok = 0;
    for c in doc["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        if name.starts_with("known_") || name.starts_with("quantity_for_notional") {
            continue; // direct calls and a tool-layer function: see the other tests
        }
        let args = c["input"]["args"].as_array().unwrap();
        let (high, low, close, volume) = (f8(&args[0]), f8(&args[1]), f8(&args[2]), f8(&args[3]));
        let kw = &c["input"]["kwargs"];
        let quantity = num(&kw["quantity"]);
        let coefficient = kw.get("coefficient").map_or(1.0, num);
        let model = kw.get("spread_model").map_or("corwin_schultz", |m| m.as_str().unwrap());
        let got = SpreadModel::parse(model)
            .and_then(|m| impact_estimate(&high, &low, &close, &volume, quantity, m, coefficient));
        if let Some(msg) = c.get("error") {
            assert_eq!(got.expect_err(name).to_string(), msg.as_str().unwrap(), "{name}");
            continue;
        }
        let got = got.unwrap_or_else(|e| panic!("{name}: {e}"));
        let w = &c["expect"];
        assert_eq!(got.days as u64, w["days"].as_u64().unwrap(), "{name}");
        let i = &w["inputs"];
        assert_eq!(got.inputs.spread_model.as_str(), i["spread_model"].as_str().unwrap(), "{name}");
        rel(&format!("{name} quantity"), got.inputs.quantity, num(&i["quantity"]));
        rel(
            &format!("{name} adv"),
            got.inputs.average_daily_volume,
            num(&i["average_daily_volume"]),
        );
        rel(&format!("{name} participation"), got.inputs.participation, num(&i["participation"]));
        rel(&format!("{name} sigma"), got.inputs.daily_sigma, num(&i["daily_sigma"]));
        rel(&format!("{name} last close"), got.inputs.last_close, num(&i["last_close"]));
        rel(
            &format!("{name} spread bps"),
            got.inputs.estimated_spread_bps,
            num(&i["estimated_spread_bps"]),
        );
        rel(
            &format!("{name} coefficient"),
            got.inputs.impact_coefficient,
            num(&i["impact_coefficient"]),
        );
        rel(&format!("{name} half spread"), got.half_spread_bps, num(&w["half_spread_bps"]));
        rel(&format!("{name} impact"), got.impact_bps, num(&w["impact_bps"]));
        rel(&format!("{name} cost bps"), got.one_way_cost_bps, num(&w["one_way_cost_bps"]));
        rel(
            &format!("{name} per unit"),
            got.one_way_cost_per_unit,
            num(&w["one_way_cost_per_unit"]),
        );
        rel(&format!("{name} total"), got.one_way_cost_total, num(&w["one_way_cost_total"]));
        assert_eq!(got.within_model_range, w["within_model_range"].as_bool().unwrap(), "{name}");
        ok += 1;
    }
    // every fixture case that is neither a direct call nor tool-layer arithmetic and is not a
    // refusal: corwin_schultz, roll_on_random_walk, roll_with_bounce, coefficient_two,
    // beyond_model_range, at_min_days, flat_bars, constant_price, negative_two_day_estimates
    assert_eq!(ok, 9, "value cases replayed");
}

#[test]
fn flat_bars_have_no_spread() {
    let doc = extract();
    let c = doc["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "known_corwin_schultz_flat")
        .unwrap();
    let args = c["input"]["args"].as_array().unwrap();
    let got = corwin_schultz(&f8(&args[0]), &f8(&args[1]));
    assert_eq!(got, num(&c["known"]["spread"]));
    assert_eq!(got, 0.0);
}

#[test]
fn alternating_prices_give_the_exact_roll_spread() {
    let doc = extract();
    let c = doc["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "known_roll_alternating")
        .unwrap();
    let close = f8(&c["input"]["args"][0]);
    rel("roll", roll(&close).expect("negative covariance"), num(&c["known"]["roll"]));
    // a straight line has a zero first-order autocovariance of changes: no estimate
    let ramp: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();
    assert_eq!(roll(&ramp), None);
}

#[test]
fn spread_model_names_round_trip() {
    for m in [SpreadModel::CorwinSchultz, SpreadModel::Roll] {
        assert_eq!(SpreadModel::parse(m.as_str()).unwrap(), m);
    }
    assert_eq!(
        SpreadModel::parse("both").unwrap_err().to_string(),
        "spread_model is corwin_schultz or roll"
    );
}
