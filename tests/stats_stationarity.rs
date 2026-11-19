//! ADF and KPSS replayed against statsmodels (`tests/python/oracle/fixtures/stats/
//! stats_stationarity.json.gz`, cut into `tests/oracle/stats_stationarity_extract.json` by
//! `tests/oracle/extract_stats_stationarity_coint.py`).
//!
//! Bars: the lag exact; the statistic and the p-value 1e-8 (absolute below 1, relative above);
//! the critical values and the table lookups 1e-12.
//!
//! Three inputs are conditioning stress tests on which the oracle is itself inexact, and are held
//! to what is true of them rather than to the oracle's digits (see `ORACLE_INEXACT`).

use raptorbt::stats::stationarity::*;
use raptorbt::stats::tables;
use raptorbt::stats::StatsError;
use serde_json::Value;

const EXTRACT: &str = include_str!("oracle/stats_stationarity_extract.json");

/// Cases whose statsmodels answer is limited by the conditioning of the regression, not by the
/// algorithm, and whose digits move with the LAPACK build the oracle ran on. With
/// `x = 100 + 1e-9 e` the oracle's t-statistic is 2.8e-6 away from the exact one (50-digit
/// arithmetic: -12.3859760928 against the oracle's -12.3859789124; 1.1e-6 on another build);
/// with `1e-8` scale noise the oracle moves by 7e-10 of the statistic between builds; with `x = 1e9 + e` its lag and statistic are numerical noise
/// (lag 17, statistic 0.185, where the exact t at lag 17 is -4.03). This port centres the series
/// first, so it is the exact answer that is returned. Held to: finite, in range, the same
/// KPSS, and for the first two the oracle's statistic to 1e-5.
const ORACLE_INEXACT: [(&str, f64); 3] =
    [("near_constant", 1e-5), ("tiny_scale", 1e-7), ("large_level", f64::INFINITY)];

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

fn cases() -> Vec<Value> {
    let v: Value = serde_json::from_str(EXTRACT).unwrap();
    v["cases"].as_array().unwrap().clone()
}

fn tables_json() -> Value {
    let v: Value = serde_json::from_str(EXTRACT).unwrap();
    v["tables"].clone()
}

fn series(case: &Value) -> Vec<f64> {
    arr(&case["input"]["args"][0])
}

fn regression(name: &str) -> Regression {
    match name {
        "n" => Regression::N,
        "c" => Regression::C,
        "ct" => Regression::Ct,
        "ctt" => Regression::Ctt,
        other => panic!("regression {other}"),
    }
}

fn inexact_tolerance(name: &str) -> Option<f64> {
    ORACLE_INEXACT.iter().find(|(n, _)| *n == name).map(|(_, t)| *t)
}

fn crit_keys<const N: usize>(v: &Value, keys: [&str; N]) -> [f64; N] {
    keys.map(|k| scalar(&v[k]))
}

/// Every case with a library answer: the ADF lag search and the KPSS test.
#[test]
fn adf_and_kpss_replay_statsmodels() {
    let mut replayed = 0;
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let lib = &case["lib"];
        // the library-only cases (fixed lags) carry a flat `lib` and are replayed below
        if lib.is_null() || lib["adf"].is_null() || lib["adf"].get("error").is_some() {
            continue;
        }
        let x = series(&case);
        let lib_adf = &lib["adf"];
        let adf = adfuller(&x, Regression::C, None, true)
            .unwrap_or_else(|e| panic!("{name}: adfuller refused: {e}"));
        assert!(adf.p_value.is_finite() && (0.0..=1.0).contains(&adf.p_value), "{name}");
        match inexact_tolerance(name) {
            None => {
                assert_eq!(adf.lags as u64, lib_adf["lags"].as_u64().unwrap(), "{name}: lags");
                assert_eq!(adf.nobs as u64, lib_adf["nobs"].as_u64().unwrap(), "{name}: nobs");
                assert_eq!(adf.maxlag as u64, lib_adf["maxlag"].as_u64().unwrap(), "{name}");
                assert_close(
                    &format!("{name} adf statistic"),
                    adf.statistic,
                    scalar(&lib_adf["statistic"]),
                    1e-8,
                );
                assert_close(
                    &format!("{name} adf p"),
                    adf.p_value,
                    scalar(&lib_adf["p_value"]),
                    1e-8,
                );
                for (a, b) in adf
                    .critical_values
                    .iter()
                    .zip(crit_keys(&lib_adf["critical_values"], ADF_CRITICAL_KEYS))
                {
                    assert_close(&format!("{name} adf critical"), *a, b, 1e-12);
                }
                assert_close(
                    &format!("{name} icbest"),
                    adf.icbest.unwrap(),
                    scalar(&lib_adf["icbest"]),
                    1e-8,
                );
                for (k, aic) in adf.aic_by_lag.iter().enumerate() {
                    assert_close(
                        &format!("{name} aic at lag {k}"),
                        *aic,
                        scalar(&lib_adf["aic_by_lag"][k.to_string()]),
                        1e-8,
                    );
                }
            }
            Some(tol) if tol.is_finite() => {
                assert_close(
                    &format!("{name} adf statistic"),
                    adf.statistic,
                    scalar(&lib_adf["statistic"]),
                    tol,
                );
            }
            Some(_) => {}
        }

        let lib_kpss = &lib["kpss"];
        let kp = kpss(&x, KpssTrend::Level, KpssLags::Auto)
            .unwrap_or_else(|e| panic!("{name}: kpss refused: {e}"));
        assert_eq!(kp.lags as u64, lib_kpss["lags"].as_u64().unwrap(), "{name}: kpss lags");
        assert_close(
            &format!("{name} kpss statistic"),
            kp.statistic,
            scalar(&lib_kpss["statistic"]),
            1e-8,
        );
        assert_close(&format!("{name} kpss p"), kp.p_value, scalar(&lib_kpss["p_value"]), 1e-8);
        assert_eq!(kp.p_value_clipped, lib_kpss["clipped"].as_bool().unwrap(), "{name}: clipped");
        for (a, b) in kp
            .critical_values
            .iter()
            .zip(crit_keys(&lib_kpss["critical_values"], KPSS_CRITICAL_KEYS))
        {
            assert_close(&format!("{name} kpss critical"), *a, b, 1e-12);
        }
        replayed += 1;
    }
    // every non-library case but `constant` (a refusal): 14 series, 13 with a library answer
    assert_eq!(replayed, 13);
}

/// The reference's whole answer: the reading and the two decisions.
#[test]
fn stationarity_reading_matches_the_reference() {
    let mut checked = 0;
    let mut inexact = 0;
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let Some(ok) = case["reference"].get("ok") else { continue };
        if case["kind"] == "library" {
            continue;
        }
        if inexact_tolerance(name).is_some() {
            inexact += 1;
            continue;
        }
        let x = series(&case);
        let got = stationarity(&x, ALPHA).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(got.n as u64, ok["n"].as_u64().unwrap(), "{name}");
        assert_eq!(got.reading.as_str(), ok["reading"].as_str().unwrap(), "{name}: reading");
        assert_eq!(
            got.rejects_unit_root,
            ok["adf"]["rejects_unit_root"].as_bool().unwrap(),
            "{name}"
        );
        assert_eq!(
            got.rejects_stationarity,
            ok["kpss"]["rejects_stationarity"].as_bool().unwrap(),
            "{name}"
        );
        assert_eq!(got.adf.lags as u64, ok["adf"]["lags"].as_u64().unwrap(), "{name}");
        assert_eq!(got.kpss.lags as u64, ok["kpss"]["lags"].as_u64().unwrap(), "{name}");
        assert_eq!(
            got.kpss.p_value_clipped,
            ok["kpss"]["p_value_clipped"].as_bool().unwrap(),
            "{name}"
        );
        assert_close(name, got.adf.statistic, scalar(&ok["adf"]["statistic"]), 1e-8);
        assert_close(name, got.kpss.statistic, scalar(&ok["kpss"]["statistic"]), 1e-8);
        checked += 1;
    }
    // the fixture holds 12 series the reference answers (the 14 minus `below_min_obs` and `constant`,
    // which it refuses); 3 of them are the oracle-inexact stress cases, leaving 9
    assert_eq!((checked, inexact), (9, 3));
}

/// The reference's refusals, as typed errors.
#[test]
fn refusals_are_typed() {
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let Some(err) = case["reference"].get("error") else { continue };
        let x = series(&case);
        let got = stationarity(&x, ALPHA).expect_err(name);
        match err["message"].as_str().unwrap() {
            m if m.contains("observations") => {
                assert!(
                    matches!(&got, StatsError::InsufficientObservations { found, needed, .. }
                    if *found == x.len() && *needed == MIN_OBS),
                    "{name}: {got:?}"
                );
                assert_eq!(got.code(), "insufficient_observations");
            }
            m if m.contains("constant") => {
                assert!(matches!(got, StatsError::DegenerateSeries(_)), "{name}: {got:?}");
                assert_eq!(got.code(), "degenerate_series");
            }
            m => panic!("{name}: unexpected reference refusal {m}"),
        }
    }
    let mut x = vec![1.0; 60];
    x[7] = f64::NAN;
    assert!(matches!(stationarity(&x, ALPHA), Err(StatsError::NonFinite { .. })));
}

/// The pieces with the lag search turned off: KPSS with a fixed or the legacy lag, and the ADF
/// for every regression at three fixed lags.
#[test]
fn fixed_lag_cases_replay_statsmodels() {
    let mut adf_cases = 0;
    let mut kpss_cases = 0;
    for case in cases() {
        if case["kind"] != "library" {
            continue;
        }
        let name = case["name"].as_str().unwrap();
        let x = series(&case);
        let lib = &case["lib"];
        let kwargs = &case["input"]["kwargs"];
        if let Some(rest) = name.strip_prefix("kpss_nlags_") {
            let lags = match rest {
                "legacy" => KpssLags::Legacy,
                n => KpssLags::Fixed(n.parse().unwrap()),
            };
            let kp = kpss(&x, KpssTrend::Level, lags).unwrap();
            assert_eq!(kp.lags as u64, lib["lags"].as_u64().unwrap(), "{name}");
            assert_close(name, kp.statistic, scalar(&lib["statistic"]), 1e-8);
            assert_close(name, kp.p_value, scalar(&lib["p_value"]), 1e-8);
            kpss_cases += 1;
        } else if name.starts_with("adf_fixed_lag_") {
            let reg = regression(kwargs["regression"].as_str().unwrap());
            let maxlag = kwargs["maxlag"].as_u64().unwrap() as usize;
            let adf = adfuller(&x, reg, Some(maxlag), false).unwrap();
            assert_eq!(adf.lags, maxlag, "{name}");
            assert_eq!(adf.nobs as u64, lib["nobs"].as_u64().unwrap(), "{name}");
            assert_close(name, adf.statistic, scalar(&lib["statistic"]), 1e-8);
            assert_close(name, adf.p_value, scalar(&lib["p_value"]), 1e-8);
            for (a, b) in adf
                .critical_values
                .iter()
                .zip(crit_keys(&lib["critical_values"], ADF_CRITICAL_KEYS))
            {
                assert_close(name, *a, b, 1e-12);
            }
            adf_cases += 1;
        }
    }
    assert_eq!((kpss_cases, adf_cases), (2, 12));
}

/// MacKinnon's p-value and critical values, and the KPSS interpolation, on the oracle's grids.
#[test]
fn tables_replay_statsmodels() {
    let t = tables_json();
    let mut cells = 0;
    for g in t["mackinnonp_1994"].as_array().unwrap() {
        let reg = regression(g["regression"].as_str().unwrap());
        let n = g["N"].as_u64().unwrap() as usize;
        for (s, p) in arr(&g["stat"]).iter().zip(arr(&g["p"])) {
            let got = mackinnon_p(*s, reg, n).unwrap();
            assert_close(&format!("p({s}, {}, N={n})", reg.name()), got, p, 1e-12);
            cells += 1;
        }
    }
    assert_eq!(cells, 24 * 16);
    for n in [7, 8, 12] {
        assert_eq!(mackinnon_p(-3.0, Regression::C, n), None, "N = {n}");
    }
    assert_eq!(t["mackinnonp_index_error"]["7"], "IndexError");
    assert_eq!(t["mackinnonp_index_error"]["12"], "IndexError");

    let mut crit_cells = 0;
    for g in t["mackinnoncrit_2010"].as_array().unwrap() {
        let reg = regression(g["regression"].as_str().unwrap());
        let n = g["N"].as_u64().unwrap() as usize;
        let nobs = arr(&g["nobs"]);
        let crit = arr(&g["crit"]);
        for (i, nb) in nobs.iter().enumerate() {
            let got = mackinnon_crit(reg, n, *nb).unwrap();
            for j in 0..3 {
                assert_close(
                    &format!("crit({}, N={n}, nobs={nb})", reg.name()),
                    got[j],
                    crit[i * 3 + j],
                    1e-12,
                );
                crit_cells += 1;
            }
        }
    }
    assert_eq!(crit_cells, 37 * 9 * 3);
    assert_eq!(t["mackinnoncrit_n_N2"], "IndexError");
    assert_eq!(mackinnon_crit(Regression::N, 2, 100.0), None);
    assert_eq!(mackinnon_crit(Regression::C, 13, 100.0), None);

    for g in t["kpss_p_interpolation"].as_array().unwrap() {
        let trend = if g["regression"] == "c" { KpssTrend::Level } else { KpssTrend::Trend };
        let table =
            if trend == KpssTrend::Level { tables::KPSS_CRIT_C } else { tables::KPSS_CRIT_CT };
        assert_eq!(arr(&g["critical"]), table.to_vec());
        for (s, p) in arr(&g["stat"]).iter().zip(arr(&g["p"])) {
            assert_close(&format!("kpss p({s})"), kpss_p_value(*s, trend), p, 1e-12);
        }
    }
}

/// Values worked by hand from the tables.
#[test]
fn hand_calculated_values() {
    // linear interpolation halfway between the 10% (0.347, p 0.10) and 5% (0.463, p 0.05) points
    assert_close("midpoint", kpss_p_value((0.347 + 0.463) / 2.0, KpssTrend::Level), 0.075, 1e-12);
    // on a critical value, its p-value
    assert_eq!(kpss_p_value(0.463, KpssTrend::Level), 0.05);
    assert_eq!(kpss_p_value(0.176, KpssTrend::Trend), 0.025);
    // outside the table the bound answers
    assert_eq!(kpss_p_value(0.0, KpssTrend::Level), 0.1);
    assert_eq!(kpss_p_value(5.0, KpssTrend::Level), 0.01);
    // MacKinnon: beyond the last tabulated statistic the p-value is 1, before the first 0
    assert_eq!(mackinnon_p(1000.0, Regression::C, 1), Some(1.0));
    assert_eq!(mackinnon_p(-1000.0, Regression::C, 1), Some(0.0));
    assert_eq!(mackinnon_p(f64::NEG_INFINITY, Regression::C, 6), Some(0.0));
    // the asymptotic critical values are the published ones
    assert_eq!(
        mackinnon_crit(Regression::C, 1, f64::INFINITY),
        Some([-3.43035, -2.86154, -2.56677])
    );
    // a finite sample: beta_inf + beta_1/T + beta_2/T^2 + beta_3/T^3 at T = 100 for the 1% value
    let [b0, b1, b2, b3] = tables::TAU_2010_C[0][0];
    let by_hand = b0 + b1 / 100.0 + b2 / 1e4 + b3 / 1e6;
    assert_close(
        "finite sample",
        mackinnon_crit(Regression::C, 1, 100.0).unwrap()[0],
        by_hand,
        1e-12,
    );
}

/// A constant in the regression makes the statistic invariant to the level of the series; the
/// port centres the series, so it holds exactly where the uncentred regression would lose digits.
#[test]
fn adf_does_not_depend_on_the_level() {
    let case = cases().into_iter().find(|c| c["name"] == "stationary_ar1").unwrap();
    let x = series(&case);
    let base = adfuller(&x, Regression::C, None, true).unwrap();
    for shift in [-1e6, 1e4, 1e9] {
        let shifted: Vec<f64> = x.iter().map(|v| v + shift).collect();
        let got = adfuller(&shifted, Regression::C, None, true).unwrap();
        assert_eq!(got.lags, base.lags, "shift {shift}");
        assert_close("statistic", got.statistic, base.statistic, 1e-6);
    }
}

/// A series that is an exact line makes the regression's columns collinear.
#[test]
fn an_exact_line_is_a_degenerate_series() {
    let x: Vec<f64> = (0..80).map(|i| 3.0 + 0.5 * i as f64).collect();
    let err = adfuller(&x, Regression::C, None, true).unwrap_err();
    assert!(matches!(err, StatsError::DegenerateSeries(_)), "{err:?}");
}
