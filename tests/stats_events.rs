//! The event study against the reference and numpy at 1e-10 (`oracle/stats_events_extract.json`):
//! market-model alpha and beta against the library's OLS, abnormal returns, CARs, t-statistics,
//! and the cross-sectional t with its p-value (`t.sf` via `math::special`).
//!
//! With no residual noise (`known_noise_free`) the t is undefined: a division by a residual sigma of
//! rounding size gives about 5e15. The reference and the engine both report NaN and set
//! `degenerate_fit`.

use raptorbt::stats::events::{
    civil_from_days, days_from_civil, event_study, iso_date, EventParams, EventStudy,
};
use serde_json::Value;

const TOL: f64 = 1e-10;

fn extract() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/oracle/stats_events_extract.json");
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

/// Returns are of the order of 1e-3: an absolute error of TOL * 1e-3 at most.
fn close(what: &str, got: f64, want: f64) {
    assert!((got - want).abs() <= TOL * want.abs().max(1e-3), "{what}: got {got:e}, want {want:e}");
}

fn day(iso: &str) -> i64 {
    let p: Vec<i64> = iso.split('-').map(|x| x.parse().unwrap()).collect();
    days_from_civil(p[0], p[1], p[2])
}

fn days(v: &Value) -> Vec<i64> {
    v.as_array().unwrap().iter().map(|d| day(d.as_str().unwrap())).collect()
}

fn same(name: &str, got: &EventStudy, want: &Value, lib: &Value, dates: &[i64]) {
    assert_eq!(got.n_bars as u64, want["n_bars"].as_u64().unwrap(), "{name}");
    assert_eq!(got.n_events as u64, want["n_events"].as_u64().unwrap(), "{name}");
    assert_eq!(got.event_window.from, want["event_window"]["from"].as_i64().unwrap(), "{name}");
    assert_eq!(got.event_window.to, want["event_window"]["to"].as_i64().unwrap(), "{name}");
    assert_eq!(got.event_window.periods as u64, want["event_window"]["periods"].as_u64().unwrap());
    assert_eq!(got.estimation.periods as u64, want["estimation"]["periods"].as_u64().unwrap());
    assert_eq!(got.estimation.gap as u64, want["estimation"]["gap"].as_u64().unwrap());
    close(&format!("{name} caar"), got.caar, num(&want["caar"]));

    let skipped = want["skipped"].as_array().unwrap();
    assert_eq!(got.skipped.len(), skipped.len(), "{name}: skipped");
    for (g, w) in got.skipped.iter().zip(skipped) {
        assert_eq!(iso_date(g.event), w["event"].as_str().unwrap(), "{name}");
        assert_eq!(g.reason, w["reason"].as_str().unwrap(), "{name}");
    }

    let rows = want["events"].as_array().unwrap();
    assert_eq!(got.events.len(), rows.len(), "{name}: events");
    let fitted: Vec<&Value> =
        lib["events"].as_array().unwrap().iter().filter(|e| e["skipped"] == false).collect();
    assert_eq!(fitted.len(), rows.len(), "{name}: fitted events in the library");
    for ((g, w), l) in got.events.iter().zip(rows).zip(fitted) {
        assert_eq!(iso_date(g.event), w["event"].as_str().unwrap(), "{name}");
        assert_eq!(iso_date(g.event_bar), w["event_bar"].as_str().unwrap(), "{name}");
        assert_eq!(g.event_bar, dates[l["idx"].as_u64().unwrap() as usize], "{name}");
        close(&format!("{name} alpha vs ols"), g.alpha, num(&l["alpha_ols"]));
        rel(&format!("{name} beta vs ols"), g.beta, num(&l["beta_ols"]));
        close(&format!("{name} alpha"), g.alpha, num(&w["alpha"]));
        rel(&format!("{name} beta"), g.beta, num(&w["beta"]));
        close(&format!("{name} sigma"), g.residual_sigma, num(&w["residual_sigma"]));
        close(&format!("{name} car"), g.car, num(&w["car"]));
        let ar = f8(&w["abnormal_returns"]);
        assert_eq!(g.abnormal_returns.len(), ar.len(), "{name}");
        for (a, b) in g.abnormal_returns.iter().zip(&ar) {
            close(&format!("{name} abnormal return"), *a, *b);
        }
        assert_eq!(g.degenerate_fit, w["degenerate_fit"].as_bool().unwrap(), "{name}");
        assert_eq!(g.degenerate_fit, name == "known_noise_free", "{name}");
        if g.degenerate_fit {
            assert!(g.t.is_nan() && num(&w["t"]).is_nan(), "{name}: t {} is not NaN", g.t);
        } else {
            rel(&format!("{name} t"), g.t, num(&w["t"]));
        }
    }

    match (got.caar_t, want["caar_t"].is_null()) {
        (Some(t), false) => {
            rel(&format!("{name} caar_t"), t, num(&want["caar_t"]));
            rel(&format!("{name} caar_p"), got.caar_p_value.unwrap(), num(&want["caar_p_value"]));
            rel(
                &format!("{name} share"),
                got.positive_share.unwrap(),
                num(&want["positive_share"]),
            );
        }
        (None, true) => {
            assert!(got.caar_p_value.is_none() && got.positive_share.is_none(), "{name}")
        }
        (g, w) => panic!("{name}: caar_t {g:?} against null={w}"),
    }
}

#[test]
fn replays_the_fixture() {
    let doc = extract();
    let series = &doc["series"];
    let mut ok = 0;
    for c in doc["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let input = &c["input"];
        let dates = days(&series[input["dates"].as_str().unwrap()]);
        let asset = f8(&series[input["asset"].as_str().unwrap()]);
        let market = f8(&series[input["market"].as_str().unwrap()]);
        let events = days(&input["events"]);
        let kw = &input["kwargs"];
        let get = |k: &str, default: i64| kw.get(k).map_or(default, |v| v.as_i64().unwrap());
        let (estimation, gap, pre, post) =
            (get("estimation", 120), get("gap", 5), get("pre", 1), get("post", 1));
        if gap < 0 || pre < 0 || post < 0 || estimation < 0 {
            // the argument cannot be expressed as periods: the binding refuses it before this call
            assert!(c.get("error").is_some(), "{name}");
            continue;
        }
        let params = EventParams {
            estimation: estimation as usize,
            gap: gap as usize,
            pre: pre as usize,
            post: post as usize,
        };
        let got = event_study(&dates, &asset, &market, &events, params);
        if c.get("error").is_some() {
            let err = got.expect_err(name);
            assert_eq!(err.code(), "invalid_input", "{name}: {err}");
            continue;
        }
        same(name, &got.unwrap_or_else(|e| panic!("{name}: {e}")), &c["expect"], &c["lib"], &dates);
        ok += 1;
    }
    assert!(ok >= 10, "only {ok} value cases replayed");
}

#[test]
fn the_fixture_error_messages_are_kept() {
    let doc = extract();
    let series = &doc["series"];
    for c in doc["cases"].as_array().unwrap() {
        let Some(msg) = c.get("error") else { continue };
        let name = c["name"].as_str().unwrap();
        let input = &c["input"];
        let dates = days(&series[input["dates"].as_str().unwrap()]);
        let asset = f8(&series[input["asset"].as_str().unwrap()]);
        let market = f8(&series[input["market"].as_str().unwrap()]);
        let kw = &input["kwargs"];
        let get = |k: &str, default: i64| kw.get(k).map_or(default, |v| v.as_i64().unwrap());
        let (estimation, gap, pre, post) =
            (get("estimation", 120), get("gap", 5), get("pre", 1), get("post", 1));
        if [estimation, gap, pre, post].iter().any(|v| *v < 0) {
            continue;
        }
        let params = EventParams {
            estimation: estimation as usize,
            gap: gap as usize,
            pre: pre as usize,
            post: post as usize,
        };
        let err =
            event_study(&dates, &asset, &market, &days(&input["events"]), params).unwrap_err();
        assert_eq!(err.to_string(), msg.as_str().unwrap(), "{name}");
    }
}

/// The market model fitted to noise-free returns: the exact answers the builder planted.
#[test]
fn noise_free_known_answers() {
    let doc = extract();
    let series = &doc["series"];
    let c =
        doc["cases"].as_array().unwrap().iter().find(|c| c["name"] == "known_noise_free").unwrap();
    let input = &c["input"];
    let dates = days(&series[input["dates"].as_str().unwrap()]);
    let got = event_study(
        &dates,
        &f8(&series[input["asset"].as_str().unwrap()]),
        &f8(&series[input["market"].as_str().unwrap()]),
        &days(&input["events"]),
        EventParams { estimation: 120, gap: 5, pre: 1, post: 1 },
    )
    .unwrap();
    let known = &c["known"];
    let e = &got.events[0];
    rel("alpha", e.alpha, num(&known["alpha"]));
    rel("beta", e.beta, num(&known["beta"]));
    rel("car", e.car, num(&known["car"]));
    rel(
        "abnormal return on the event",
        e.abnormal_returns[1],
        num(&known["abnormal_return_on_event"]),
    );
}

#[test]
fn caar_p_value_is_the_two_sided_t_tail() {
    let doc = extract();
    let ex = &doc["tables"]["caar_p_example"];
    let cars = f8(&ex["car"]);
    let n = cars.len() as f64;
    let mean = cars.iter().sum::<f64>() / n;
    let sd = (cars.iter().map(|c| (c - mean) * (c - mean)).sum::<f64>() / (n - 1.0)).sqrt();
    let t = mean / (sd / n.sqrt());
    rel("t", t, num(&ex["t"]));
    rel("p", raptorbt::math::special::t_two_sided_p(t.abs(), n - 1.0).unwrap(), num(&ex["p"]));
}

#[test]
fn civil_dates_round_trip() {
    assert_eq!(days_from_civil(1970, 1, 1), 0);
    assert_eq!(iso_date(days_from_civil(2024, 2, 29)), "2024-02-29");
    assert_eq!(iso_date(-1), "1969-12-31");
    let mut d = days_from_civil(1900, 1, 1);
    let end = days_from_civil(2100, 12, 31);
    let mut previous = None;
    while d <= end {
        let (y, m, day) = civil_from_days(d);
        assert_eq!(days_from_civil(y, m, day), d);
        if let Some(p) = previous {
            assert_eq!(d, p + 1);
        }
        previous = Some(d);
        d += 1;
    }
}
