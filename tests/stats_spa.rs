//! Superior predictive ability against arch 8.0.0 (`oracle/stats_spa_extract*.json`).
//!
//! Replaying arch's stored bootstrap indices reproduces its loss-differential means and
//! variances, its recentring flags, the largest simulated value of each repetition, and its three
//! p-values, at 1e-10 (the p-values are counts over the repetitions and equal exactly). With this
//! crate's own seed the p-values are Monte Carlo estimates of the same quantity: they must lie
//! within 4 standard errors of the difference of two such estimates of arch's value.

use raptorbt::stats::spa::{spa_replay, spa_test};
use raptorbt::stats::StatsError;
use serde_json::Value;

const TOL: f64 = 1e-10;

fn extracts() -> Vec<Value> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/oracle");
    let mut out = Vec::new();
    for i in 1.. {
        let name = if i == 1 {
            "stats_spa_extract.json".to_string()
        } else {
            format!("stats_spa_extract_{i}.json")
        };
        let path = format!("{dir}/{name}");
        let Ok(text) = std::fs::read_to_string(&path) else { break };
        out.push(serde_json::from_str(&text).expect("json"));
    }
    assert!(!out.is_empty(), "no extract");
    out
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

/// Absolute error against a value scaled at least by 1e-3 (returns are of that size).
fn close(what: &str, got: f64, want: f64) {
    let err = (got - want).abs();
    assert!(err <= TOL * want.abs().max(1e-3), "{what}: got {got:e}, want {want:e}");
}

fn columns(v: &Value) -> Vec<Vec<f64>> {
    v["order"].as_array().unwrap().iter().map(|n| f8(&v["columns"][n.as_str().unwrap()])).collect()
}

fn indices(v: &Value) -> Vec<Vec<u32>> {
    let width = v["width"].as_u64().unwrap() as usize;
    let hex = v["hex"].as_str().unwrap();
    let (reps, n) =
        (v["shape"][0].as_u64().unwrap() as usize, v["shape"][1].as_u64().unwrap() as usize);
    let digits = 2 * width;
    (0..reps)
        .map(|r| {
            (0..n)
                .map(|t| {
                    let at = (r * n + t) * digits;
                    u32::from_str_radix(&hex[at..at + digits], 16).unwrap()
                })
                .collect()
        })
        .collect()
}

#[test]
fn replays_arch_with_its_own_indices() {
    let mut replayed = 0;
    for doc in extracts() {
        for c in doc["cases"].as_array().unwrap() {
            let (name, lib) = (c["name"].as_str().unwrap(), &c["lib"]);
            if lib.is_null() {
                continue;
            }
            let cols = columns(&c["input"]["args"][0]);
            let bench = c["input"]["kwargs"].get("benchmark").map(f8);
            let diff: Vec<Vec<f64>> = cols
                .iter()
                .map(|col| match &bench {
                    Some(b) => col.iter().zip(b).map(|(x, y)| x - y).collect(),
                    None => col.clone(),
                })
                .collect();
            let block = lib["block"].as_u64().unwrap() as usize;
            let got = spa_replay(&diff, block, &indices(&lib["indices"])).expect(name);

            for (j, (m, v)) in
                f8(&lib["loss_diff_mean"]).iter().zip(f8(&lib["loss_diff_var"])).enumerate()
            {
                close(&format!("{name} mean[{j}]"), got.loss_diff_mean[j], *m);
                rel(&format!("{name} var[{j}]"), got.loss_diff_var[j], v);
            }
            let valid: Vec<bool> =
                lib["valid_columns"].as_array().unwrap().iter().map(|v| v == 1).collect();
            assert_eq!(got.valid, valid, "{name}: recentring flags");

            let sim = f8(&lib["max_simulated"]);
            for (r, row) in got.max_simulated.iter().enumerate() {
                for (v, x) in row.iter().enumerate() {
                    close(&format!("{name} max_simulated[{r}][{v}]"), *x, sim[r * 3 + v]);
                }
            }
            assert_eq!(got.pvalues.lower, num(&lib["p_lower"]), "{name} lower");
            assert_eq!(got.pvalues.consistent, num(&lib["p_consistent"]), "{name} consistent");
            assert_eq!(got.pvalues.upper, num(&lib["p_upper"]), "{name} upper");
            replayed += 1;
        }
    }
    assert!(replayed >= 10, "only {replayed} cases replayed");
}

fn within_mc(what: &str, got: f64, want: f64, reps: usize) {
    let r = reps as f64;
    // two estimates of one probability: the difference has variance 2 p (1 - p) / reps; one
    // repetition of slack covers p at 0 or 1
    let bound = 4.0 * (2.0 * want * (1.0 - want) / r).sqrt() + 1.0 / r;
    assert!((got - want).abs() <= bound, "{what}: got {got}, arch {want}, allowed +-{bound}");
}

#[test]
fn own_seed_agrees_with_arch_within_monte_carlo_error() {
    let mut checked = 0;
    for doc in extracts() {
        for c in doc["cases"].as_array().unwrap() {
            let (name, lib) = (c["name"].as_str().unwrap(), &c["lib"]);
            if lib.is_null() {
                continue;
            }
            let cols = columns(&c["input"]["args"][0]);
            let kw = &c["input"]["kwargs"];
            let bench = kw.get("benchmark").map(f8);
            let reps = kw["reps"].as_u64().unwrap() as usize;
            let seed = kw["seed"].as_u64().unwrap();
            let block = kw.get("block_size").map(|b| b.as_i64().unwrap());
            let got = spa_test(&cols, bench.as_deref(), reps, block, seed).expect(name);

            let want = &c["expect"];
            assert_eq!(got.mean_block as u64, lib["block"].as_u64().unwrap(), "{name} block");
            assert_eq!(got.reps, reps);
            assert_eq!(got.seed, seed);
            let order = c["input"]["args"][0]["order"].as_array().unwrap();
            assert_eq!(order[got.best_strategy].as_str(), want["best_strategy"].as_str(), "{name}");
            close(
                &format!("{name} best mean"),
                got.best_mean_excess,
                num(&want["best_mean_excess"]),
            );
            match (
                got.best_per_period_sharpe_excess,
                want["best_per_period_sharpe_excess"].is_null(),
            ) {
                (Some(s), false) => {
                    rel(&format!("{name} sharpe"), s, num(&want["best_per_period_sharpe_excess"]))
                }
                (None, true) => {}
                (g, w) => panic!("{name}: sharpe {g:?}, arch null: {w}"),
            }
            within_mc(&format!("{name} lower"), got.p_value_spa_lower, num(&lib["p_lower"]), reps);
            within_mc(
                &format!("{name} consistent"),
                got.p_value_spa_consistent,
                num(&lib["p_consistent"]),
                reps,
            );
            within_mc(
                &format!("{name} upper"),
                got.p_value_white_reality_check,
                num(&lib["p_upper"]),
                reps,
            );
            checked += 1;
        }
    }
    assert!(checked >= 10, "only {checked} cases checked");
}

#[test]
fn refusals_replay_the_reference() {
    for doc in extracts() {
        for c in doc["cases"].as_array().unwrap() {
            let Some(msg) = c.get("error") else { continue };
            let name = c["name"].as_str().unwrap();
            let cols = columns(&c["input"]["args"][0]);
            let kw = &c["input"]["kwargs"];
            let bench = kw.get("benchmark").map(f8);
            let reps = kw.get("reps").map_or(1000, |r| r.as_u64().unwrap() as usize);
            let seed = kw.get("seed").map_or(0, |s| s.as_u64().unwrap());
            let block = kw.get("block_size").map(|b| b.as_i64().unwrap());
            let err = spa_test(&cols, bench.as_deref(), reps, block, seed).expect_err(name);
            let want_code = if msg.as_str().unwrap().contains("this statistic needs") {
                "insufficient_observations"
            } else {
                "invalid_input"
            };
            assert_eq!(err.code(), want_code, "{name}: {err}");
        }
    }
}

fn synthetic(k: usize, rows: usize) -> Vec<Vec<f64>> {
    (0..k)
        .map(|j| {
            (0..rows)
                .map(|t| (((t * 31 + j * 17 + (t * j) % 13) % 97) as f64 - 46.0) / 1000.0)
                .collect()
        })
        .collect()
}

#[test]
fn caps_the_number_of_strategies() {
    assert!(spa_test(&synthetic(200, 60), None, 200, None, 1).is_ok());
    assert!(matches!(
        spa_test(&synthetic(201, 60), None, 200, None, 1),
        Err(StatsError::Invalid(_))
    ));
}

#[test]
fn the_answer_does_not_depend_on_the_thread_count() {
    let cols = synthetic(12, 150);
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| spa_test(&cols, None, 400, None, 99).unwrap())
    };
    let one = run(1);
    assert_eq!(one, run(3));
    assert_eq!(one, run(8));
}

#[test]
fn a_constant_excess_is_refused_not_reported_as_p_zero() {
    // Zero excess against a cash benchmark, and strategies equal to a non-zero benchmark: no
    // resample can exceed the observed mean, which used to print every p-value as 0.
    let zeros = vec![vec![0.0; 300]; 2];
    let e = spa_test(&zeros, None, 200, None, 1).unwrap_err();
    assert_eq!(e.code(), "degenerate_series", "{e}");
    let bench: Vec<f64> = (0..300).map(|t| ((t * 37 % 101) as f64 - 50.0) / 5000.0).collect();
    let same = vec![bench.clone(), bench.clone()];
    let e = spa_test(&same, Some(&bench), 200, None, 1).unwrap_err();
    assert_eq!(e.code(), "degenerate_series", "{e}");
    // One column that varies is enough to test.
    let mixed = vec![bench.clone(), vec![0.0; 300]];
    assert!(spa_test(&mixed, Some(&bench), 200, None, 1).is_ok());
    // The replay keeps its pin to arch and takes constant columns as given.
    let idx: Vec<Vec<u32>> = vec![(0..300).collect(); 3];
    assert!(spa_replay(&zeros, 6, &idx).is_ok());
}
