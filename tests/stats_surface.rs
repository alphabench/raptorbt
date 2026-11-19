//! The parameter-sensitivity surface against the reference (`oracle/stats_surface_extract.json`),
//! 1e-12 or exact: smoothness, the best cell and its plateau share, the robust pick, the verdict,
//! the cell table and the matrix. The conventions the doc states are pinned by the tied-best and
//! isolated-best cases.

use raptorbt::stats::surface::{sensitivity_surface, Surface};
use serde_json::Value;

const TOL: f64 = 1e-12;

fn extract() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/oracle/stats_surface_extract.json");
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
    let err = (got - want).abs();
    assert!(err <= TOL * want.abs().max(1.0), "{what}: got {got:e}, want {want:e}");
}

fn opt(what: &str, got: Option<f64>, want: &Value) {
    match (got, want.is_null()) {
        (Some(g), false) => rel(what, g, num(want)),
        (None, true) => {}
        (g, w) => panic!("{what}: {g:?} against null={w}"),
    }
}

fn params(got: &[f64], names: &[String], want: &Value, what: &str) {
    for (g, n) in got.iter().zip(names) {
        rel(&format!("{what} {n}"), *g, num(&want[n]));
    }
}

fn inputs(c: &Value) -> (Vec<String>, Vec<(Vec<f64>, f64)>) {
    let args = c["input"]["args"].as_array().unwrap();
    let names =
        args[0].as_array().unwrap().iter().map(|n| n.as_str().unwrap().to_string()).collect();
    let cells = args[1]
        .as_array()
        .unwrap()
        .iter()
        .map(|cell| {
            let vals = cell[0].as_array().map_or_else(Vec::new, |v| v.iter().map(num).collect());
            (vals, num(&cell[1]))
        })
        .collect();
    (names, cells)
}

fn same(name: &str, got: &Surface, names: &[String], w: &Value) {
    assert_eq!(got.n_cells as u64, w["n_cells"].as_u64().unwrap(), "{name}");
    assert_eq!(got.grid_cells as u64, w["grid_cells"].as_u64().unwrap(), "{name}");
    rel(&format!("{name} coverage"), got.coverage, num(&w["coverage"]));
    rel(&format!("{name} min"), got.metric_min, num(&w["metric_min"]));
    rel(&format!("{name} max"), got.metric_max, num(&w["metric_max"]));
    rel(&format!("{name} smoothness"), got.smoothness, num(&w["smoothness"]));
    assert_eq!(got.verdict.as_str(), w["verdict"].as_str().unwrap(), "{name}");
    for (i, n) in names.iter().enumerate() {
        assert_eq!(got.axes[i], f8(&w["axes"][n]), "{name} axis {n}");
    }
    let b = &w["best"];
    params(&got.best.params, names, &b["params"], &format!("{name} best"));
    rel(&format!("{name} best metric"), got.best.metric, num(&b["metric"]));
    opt(&format!("{name} best neighbour mean"), got.best.neighbour_mean, &b["neighbour_mean"]);
    assert_eq!(got.best.neighbours as u64, b["neighbours"].as_u64().unwrap(), "{name}");
    rel(&format!("{name} plateau share"), got.best.plateau_share, num(&b["plateau_share"]));
    let r = &w["robust_pick"];
    params(&got.robust_pick.params, names, &r["params"], &format!("{name} robust"));
    rel(&format!("{name} robust metric"), got.robust_pick.metric, num(&r["metric"]));
    rel(&format!("{name} robust nb"), got.robust_pick.neighbour_mean, num(&r["neighbour_mean"]));

    let rows = w["cells"].as_array().unwrap();
    assert_eq!(got.cells.len(), rows.len(), "{name}");
    for (g, row) in got.cells.iter().zip(rows) {
        params(&g.params, names, &row["params"], &format!("{name} cell"));
        rel(&format!("{name} cell metric"), g.metric, num(&row["metric"]));
        opt(&format!("{name} cell nb"), g.neighbour_mean, &row["neighbour_mean"]);
        opt(&format!("{name} cell lift"), g.lift_over_neighbours, &row["lift_over_neighbours"]);
        assert_eq!(g.neighbours as u64, row["neighbours"].as_u64().unwrap(), "{name}");
    }
    match (&got.matrix, w.get("matrix")) {
        (None, None) => {}
        (Some(m), Some(wm)) => {
            let values = wm["values"].as_array().unwrap();
            assert_eq!(m.values.len(), values.len(), "{name}");
            for (gr, wr) in m.values.iter().zip(values) {
                // a row of floats is one packed f8 array; a row with unrun cells is a list
                let wr: Vec<Option<f64>> = if wr.get("f8").is_some() {
                    f8(wr).into_iter().map(Some).collect()
                } else {
                    wr.as_array().unwrap().iter().map(|x| (!x.is_null()).then(|| num(x))).collect()
                };
                assert_eq!(gr.len(), wr.len(), "{name}");
                for (g, w) in gr.iter().zip(&wr) {
                    match (g, w) {
                        (Some(g), Some(w)) => rel(&format!("{name} matrix"), *g, *w),
                        (None, None) => {}
                        (g, w) => panic!("{name}: matrix cell {g:?} against {w:?}"),
                    }
                }
            }
        }
        (g, w) => panic!("{name}: matrix {} against {}", g.is_some(), w.is_some()),
    }
}

#[test]
fn replays_the_fixture() {
    let doc = extract();
    let mut ok = 0;
    for c in doc["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let (names, cells) = inputs(c);
        let got = sensitivity_surface(&names, &cells);
        if c.get("error").is_some() {
            assert_eq!(got.expect_err(name).code(), "invalid_input", "{name}");
            continue;
        }
        let got = got.unwrap_or_else(|e| panic!("{name}: {e}"));
        same(name, &got, &names, &c["expect"]);
        if let Some(known) = c.get("known") {
            rel(&format!("{name} known best"), got.best.metric, num(&known["best_metric"]));
            rel(
                &format!("{name} known nb"),
                got.best.neighbour_mean.unwrap(),
                num(&known["neighbour_mean_of_best"]),
            );
            rel(
                &format!("{name} known share"),
                got.best.plateau_share,
                num(&known["plateau_share"]),
            );
            rel(&format!("{name} known smoothness"), got.smoothness, num(&known["smoothness"]));
            assert_eq!(got.verdict.as_str(), known["verdict"].as_str().unwrap(), "{name}");
        }
        ok += 1;
    }
    assert!(ok >= 10, "only {ok} value cases replayed");
}

#[test]
fn the_first_of_tied_best_cells_in_input_order_wins() {
    let doc = extract();
    let find =
        |n: &str| doc["cases"].as_array().unwrap().iter().find(|c| c["name"] == n).unwrap().clone();
    let (fwd, rev) = (find("tied_best_cells"), find("tied_best_cells_reversed"));
    let run = |c: &Value| {
        let (names, cells) = inputs(c);
        sensitivity_surface(&names, &cells).unwrap()
    };
    let (a, b) = (run(&fwd), run(&rev));
    assert_eq!(a.best.metric, b.best.metric);
    assert_ne!(a.best.params, b.best.params, "the order of the cells picks the winner");
}

#[test]
fn caps_the_number_of_cells() {
    let names = ["x".to_string(), "y".to_string()];
    let grid = |n: usize| -> Vec<(Vec<f64>, f64)> {
        (0..n).map(|i| (vec![(i % 50) as f64, (i / 50) as f64], (i * 7 % 31) as f64)).collect()
    };
    assert!(sensitivity_surface(&names, &grid(2000)).is_ok());
    assert_eq!(sensitivity_surface(&names, &grid(2001)).unwrap_err().code(), "invalid_input");
}
