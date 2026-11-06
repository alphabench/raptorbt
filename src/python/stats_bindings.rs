//! Python bindings for the statistics in `crate::stats`. Each function takes numpy arrays and
//! plain values, runs the statistic with the interpreter released, and returns a dict whose keys
//! are the result fields, at full precision (NaN stays NaN; the caller rounds). The English that
//! goes with a result (`assumptions`, `meaning`, ...) and the argument checks that need Python
//! objects (dates, mappings) live in `raptorbt.stats`, which is the public surface.
//!
//! A refusal is a structured `RaptorBTError`: the code comes from `StatsError::code()`.

use std::borrow::Cow;

use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use super::errors::Fail;
use super::numpy_bridge::numpy_cow_f64;
use crate::stats::coint::{self, EngleGranger};
use crate::stats::events::{self, iso_date, EventParams};
use crate::stats::garch::{self, Dist, GarchError, GarchOptions, Model};
use crate::stats::hac;
use crate::stats::hmm::{self, HmmError, HmmOptions};
use crate::stats::hurst;
use crate::stats::impact::{self, SpreadModel};
use crate::stats::pbo;
use crate::stats::pca;
use crate::stats::quality::{self, AuditRules, Bars, Gap};
use crate::stats::spa;
use crate::stats::stationarity::{self, ADF_CRITICAL_KEYS, KPSS_CRITICAL_KEYS};
use crate::stats::surface;
use crate::stats::StatsError;

type Dict<'py> = Bound<'py, PyDict>;

// ------------------------------------------------------------------------------------ errors

/// The documentation domain every statistics refusal points at.
const SKILL: &str = "statistics";

fn fail(operation: &str, e: StatsError) -> PyErr {
    Fail::new(e.code(), e.to_string()).op(operation).skill(SKILL).into()
}

fn invalid(operation: &str, message: impl Into<String>) -> PyErr {
    Fail::new("invalid_input", message).op(operation).skill(SKILL).into()
}

fn garch_fail(operation: &str, e: GarchError) -> PyErr {
    let message = e.to_string();
    fail(
        operation,
        match e {
            GarchError::BadHorizon { .. } => StatsError::Invalid(message),
            GarchError::InsufficientObservations { found, needed } => {
                StatsError::InsufficientObservations { what: "returns".into(), found, needed }
            }
            GarchError::NonFinite => StatsError::NonFinite { what: "returns".into() },
            GarchError::DegenerateSeries => StatsError::DegenerateSeries(message),
            GarchError::DidNotConverge => StatsError::DidNotConverge(message),
        },
    )
}

fn hmm_fail(operation: &str, e: HmmError) -> PyErr {
    let message = e.to_string();
    fail(
        operation,
        match e {
            HmmError::BadRegimes { .. } => StatsError::Invalid(message),
            HmmError::InsufficientObservations { found, needed } => {
                StatsError::InsufficientObservations { what: "returns".into(), found, needed }
            }
            HmmError::NonFinite => StatsError::NonFinite { what: "returns".into() },
            HmmError::DegenerateSeries | HmmError::DegenerateFit => {
                StatsError::DegenerateSeries(message)
            }
            HmmError::DidNotConverge => StatsError::DidNotConverge(message),
        },
    )
}

/// A count argument: a negative Python int is the same refusal as one past the range.
fn count(operation: &str, value: i64, message: &str) -> PyResult<usize> {
    usize::try_from(value).map_err(|_| invalid(operation, message))
}

// ---------------------------------------------------------------------------------- builders

fn keyed<'py>(py: Python<'py>, keys: &[&str], values: &[f64]) -> PyResult<Dict<'py>> {
    let d = PyDict::new(py);
    for (k, v) in keys.iter().zip(values) {
        d.set_item(*k, *v)?;
    }
    Ok(d)
}

fn named<'py, N: AsRef<str>>(py: Python<'py>, names: &[N], values: &[f64]) -> PyResult<Dict<'py>> {
    let d = PyDict::new(py);
    for (k, v) in names.iter().zip(values) {
        d.set_item(k.as_ref(), *v)?;
    }
    Ok(d)
}

fn columns<'a, 'py>(arrays: &'a [PyReadonlyArray1<'py, f64>]) -> Vec<Cow<'a, [f64]>> {
    arrays.iter().map(numpy_cow_f64).collect()
}

// ------------------------------------------------------------------------------ stationarity

#[pyfunction]
#[pyo3(signature = (series, alpha))]
fn stats_stationarity<'py>(
    py: Python<'py>,
    series: PyReadonlyArray1<'py, f64>,
    alpha: f64,
) -> PyResult<Dict<'py>> {
    const OP: &str = "raptorbt.stats.stationarity";
    let x = numpy_cow_f64(&series);
    let r = py.detach(|| stationarity::stationarity(&x, alpha)).map_err(|e| fail(OP, e))?;
    let adf = PyDict::new(py);
    adf.set_item("statistic", r.adf.statistic)?;
    adf.set_item("p_value", r.adf.p_value)?;
    adf.set_item("lags", r.adf.lags)?;
    adf.set_item("critical_values", keyed(py, &ADF_CRITICAL_KEYS, &r.adf.critical_values)?)?;
    adf.set_item("rejects_unit_root", r.rejects_unit_root)?;
    adf.set_item("nobs", r.adf.nobs)?;
    adf.set_item("maxlag", r.adf.maxlag)?;
    adf.set_item("icbest", r.adf.icbest)?;
    let kpss = PyDict::new(py);
    kpss.set_item("statistic", r.kpss.statistic)?;
    kpss.set_item("p_value", r.kpss.p_value)?;
    kpss.set_item("p_value_clipped", r.kpss.p_value_clipped)?;
    kpss.set_item("lags", r.kpss.lags)?;
    kpss.set_item("critical_values", keyed(py, &KPSS_CRITICAL_KEYS, &r.kpss.critical_values)?)?;
    kpss.set_item("rejects_stationarity", r.rejects_stationarity)?;
    let d = PyDict::new(py);
    d.set_item("n", r.n)?;
    d.set_item("alpha", r.alpha)?;
    d.set_item("reading", r.reading.as_str())?;
    d.set_item("adf", adf)?;
    d.set_item("kpss", kpss)?;
    Ok(d)
}

// ------------------------------------------------------------------------------ cointegration

fn engle_granger_dict<'py, N: AsRef<str>>(
    py: Python<'py>,
    names: &[N],
    eg: &EngleGranger,
) -> PyResult<Dict<'py>> {
    let d = PyDict::new(py);
    d.set_item("dependent", names[0].as_ref())?;
    d.set_item("statistic", eg.statistic)?;
    d.set_item("p_value", eg.p_value)?;
    d.set_item("critical_values", keyed(py, &coint::CRITICAL_KEYS, &eg.critical_values)?)?;
    d.set_item("cointegrated", eg.cointegrated)?;
    d.set_item("intercept", eg.intercept)?;
    d.set_item("hedge_ratios", named(py, &names[1..], &eg.hedge_ratios)?)?;
    d.set_item("spread_mean", eg.spread_mean)?;
    d.set_item("spread_std", eg.spread_std)?;
    d.set_item("spread_last_z", eg.spread_last_z)?;
    d.set_item("half_life_periods", eg.half_life_periods)?;
    d.set_item("adf_lags", eg.adf_lags)?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (names, series, alpha))]
fn stats_cointegration<'py>(
    py: Python<'py>,
    names: Vec<String>,
    series: Vec<PyReadonlyArray1<'py, f64>>,
    alpha: f64,
) -> PyResult<Dict<'py>> {
    const OP: &str = "raptorbt.stats.cointegration";
    let cols = columns(&series);
    let refs: Vec<&[f64]> = cols.iter().map(|c| c.as_ref()).collect();
    let r = py.detach(|| coint::cointegration(&refs, alpha)).map_err(|e| fail(OP, e))?;
    let jo = PyDict::new(py);
    jo.set_item("rank_at_5pct", r.johansen.rank_at_5pct)?;
    jo.set_item("trace_statistics", &r.johansen.trace_statistics)?;
    jo.set_item("trace_critical_5pct", &r.johansen.trace_critical_5pct)?;
    jo.set_item("cointegrating_vector", named(py, &names, &r.johansen.cointegrating_vector)?)?;
    let d = PyDict::new(py);
    d.set_item("n", r.n)?;
    d.set_item("series", &names)?;
    d.set_item("alpha", r.alpha)?;
    d.set_item("engle_granger", engle_granger_dict(py, &names, &r.engle_granger)?)?;
    d.set_item("johansen", jo)?;
    Ok(d)
}

// ---------------------------------------------------------------------------------- memory

#[pyfunction]
#[pyo3(signature = (increments,))]
fn stats_hurst<'py>(
    py: Python<'py>,
    increments: PyReadonlyArray1<'py, f64>,
) -> PyResult<Dict<'py>> {
    const OP: &str = "raptorbt.stats.hurst";
    let x = numpy_cow_f64(&increments);
    let r = py.detach(|| hurst::hurst(&x)).map_err(|e| fail(OP, e))?;
    let rs = PyDict::new(py);
    rs.set_item("hurst", r.rescaled_range.hurst)?;
    rs.set_item("raw_slope", r.rescaled_range.raw_slope)?;
    rs.set_item("std_error", r.rescaled_range.std_error)?;
    rs.set_item("scales", &r.rescaled_range.scales)?;
    let dfa = PyDict::new(py);
    dfa.set_item("hurst", r.dfa.hurst)?;
    dfa.set_item("std_error", r.dfa.std_error)?;
    dfa.set_item("scales", &r.dfa.scales)?;
    let d = PyDict::new(py);
    d.set_item("n", r.n)?;
    d.set_item("rescaled_range", rs)?;
    d.set_item("dfa", dfa)?;
    Ok(d)
}

// -------------------------------------------------------------------------------- volatility

#[pyfunction]
#[pyo3(signature = (returns, model, dist, horizon, periods_per_year))]
fn stats_garch<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray1<'py, f64>,
    model: &str,
    dist: &str,
    horizon: i64,
    periods_per_year: Option<u32>,
) -> PyResult<Dict<'py>> {
    const OP: &str = "raptorbt.stats.garch";
    let (model, dist) = match (model, dist) {
        ("garch", "normal") => (Model::Garch, Dist::Normal),
        ("garch", "t") => (Model::Garch, Dist::StudentT),
        ("gjr", "normal") => (Model::Gjr, Dist::Normal),
        ("gjr", "t") => (Model::Gjr, Dist::StudentT),
        _ => return Err(invalid(OP, "model is garch or gjr, dist is normal or t")),
    };
    let message = format!("horizon must be 1 to {}", garch::MAX_HORIZON);
    let opts = GarchOptions {
        model,
        dist,
        horizon: count(OP, horizon, &message)?,
        periods_per_year: periods_per_year.filter(|p| *p > 0),
    };
    let x = numpy_cow_f64(&returns);
    let r = py.detach(|| garch::garch(&x, &opts)).map_err(|e| garch_fail(OP, e))?;
    let p = &r.parameters;
    let params = PyDict::new(py);
    params.set_item("mu", p.mu)?;
    params.set_item("omega", p.omega)?;
    params.set_item("alpha", p.alpha)?;
    params.set_item("gamma", p.gamma)?;
    params.set_item("beta", p.beta)?;
    params.set_item("nu", p.nu)?;
    let d = PyDict::new(py);
    d.set_item("n", r.n)?;
    d.set_item("model", r.model)?;
    d.set_item("distribution", r.distribution)?;
    d.set_item("converged", r.converged)?;
    d.set_item("parameters", params)?;
    if let Some(se) = &r.std_errors {
        let s = PyDict::new(py);
        s.set_item("mu", se.mu)?;
        s.set_item("omega", se.omega)?;
        s.set_item("alpha", se.alpha)?;
        s.set_item("gamma", se.gamma)?;
        s.set_item("beta", se.beta)?;
        s.set_item("nu", se.nu)?;
        d.set_item("std_errors", s)?;
    } else {
        d.set_item("std_errors", py.None())?;
    }
    d.set_item("covariance_robust", &r.covariance_robust)?;
    d.set_item("parameter_names", r.parameter_names.clone())?;
    d.set_item("persistence", r.persistence)?;
    d.set_item("stationary_variance", r.stationary_variance)?;
    d.set_item("half_life_periods", r.half_life_periods)?;
    d.set_item("unconditional_vol_per_period", r.unconditional_vol_per_period)?;
    d.set_item("log_likelihood", r.log_likelihood)?;
    d.set_item("aic", r.aic)?;
    d.set_item("last_vol_per_period", r.last_vol_per_period)?;
    d.set_item("forecast_vol_per_period", &r.forecast_vol_per_period)?;
    d.set_item("forecast_cumulative_vol", r.forecast_cumulative_vol)?;
    d.set_item("recent_conditional_vol", &r.recent_conditional_vol)?;
    d.set_item("rescale", r.rescale)?;
    if let Some(v) = r.last_vol_annualised {
        d.set_item("last_vol_annualised", v)?;
    }
    if let Some(v) = &r.forecast_vol_annualised {
        d.set_item("forecast_vol_annualised", v)?;
    }
    if let Some(v) = r.periods_per_year {
        d.set_item("periods_per_year", v)?;
    }
    Ok(d)
}

// ----------------------------------------------------------------------------------- regimes

#[pyfunction]
#[pyo3(signature = (returns, regimes, min_support_bars, min_duration_periods, series))]
fn stats_regimes<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray1<'py, f64>,
    regimes: i64,
    min_support_bars: f64,
    min_duration_periods: f64,
    series: bool,
) -> PyResult<Dict<'py>> {
    const OP: &str = "raptorbt.stats.markov_regimes";
    let opts = HmmOptions {
        regimes: count(OP, regimes, "k must be 2 or 3 regimes")?,
        min_support_bars,
        min_duration_periods,
    };
    let x = numpy_cow_f64(&returns);
    let r = py.detach(|| hmm::markov_regimes(&x, &opts)).map_err(|e| hmm_fail(OP, e))?;
    let stats = PyList::empty(py);
    for s in &r.regime_stats {
        let row = PyDict::new(py);
        row.set_item("regime", s.regime)?;
        row.set_item("mean_per_period", s.mean_per_period)?;
        row.set_item("vol_per_period", s.vol_per_period)?;
        row.set_item("share_of_periods", s.share_of_periods)?;
        row.set_item("stay_probability", s.stay_probability)?;
        row.set_item("expected_duration_periods", s.expected_duration_periods)?;
        row.set_item("support_bars", s.support_bars)?;
        row.set_item("low_support", s.low_support)?;
        stats.append(row)?;
    }
    let timeline = PyList::empty(py);
    for t in &r.timeline {
        let row = PyDict::new(py);
        row.set_item("regime", t.regime)?;
        row.set_item("start_index", t.start_index)?;
        row.set_item("end_index", t.end_index)?;
        row.set_item("periods", t.periods)?;
        timeline.append(row)?;
    }
    let d = PyDict::new(py);
    d.set_item("n", r.n)?;
    d.set_item("regimes", r.regimes)?;
    d.set_item("converged", r.converged)?;
    d.set_item("regime_stats", stats)?;
    d.set_item("low_support_regimes", &r.low_support_regimes)?;
    d.set_item("transition_matrix", &r.transition_matrix)?;
    d.set_item("current_regime", r.current_regime)?;
    d.set_item("current_probabilities", &r.current_probabilities)?;
    d.set_item("recent_probabilities", &r.recent_probabilities)?;
    d.set_item("recent_filtered_probabilities", &r.recent_filtered_probabilities)?;
    d.set_item("timeline_total", r.timeline_total)?;
    d.set_item("timeline", timeline)?;
    d.set_item("log_likelihood", r.log_likelihood)?;
    d.set_item("aic", r.aic)?;
    d.set_item("smoothed_is_causal", r.smoothed_is_causal)?;
    d.set_item("filtered_is_causal", r.filtered_is_causal)?;
    d.set_item("start_log_likelihoods", &r.start_log_likelihoods)?;
    d.set_item("winning_start", r.winning_start)?;
    if series {
        d.set_item("smoothed_probabilities", PyArray2::from_vec2(py, &r.smoothed_probabilities)?)?;
        d.set_item("filtered_probabilities", PyArray2::from_vec2(py, &r.filtered_probabilities)?)?;
    }
    Ok(d)
}

// ---------------------------------------------------------------------------------- inference

#[pyfunction]
#[pyo3(signature = (returns, lags))]
fn stats_hac<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray1<'py, f64>,
    lags: Option<i64>,
) -> PyResult<Dict<'py>> {
    const OP: &str = "raptorbt.stats.hac_tstat";
    let x = numpy_cow_f64(&returns);
    let r = py.detach(|| hac::hac_tstat(&x, lags)).map_err(|e| fail(OP, e))?;
    let d = PyDict::new(py);
    d.set_item("n", r.n)?;
    d.set_item("mean", r.mean)?;
    d.set_item("hac_std_error", r.hac_std_error)?;
    d.set_item("hac_t", r.hac_t)?;
    d.set_item("hac_p_value", r.hac_p_value)?;
    d.set_item("naive_t", r.naive_t)?;
    d.set_item("naive_p_value", r.naive_p_value)?;
    d.set_item("lags", r.lags)?;
    d.set_item("per_period_sharpe", r.per_period_sharpe)?;
    d.set_item("lag1_autocorrelation", r.lag1_autocorrelation)?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (strategies, splits))]
fn stats_pbo<'py>(
    py: Python<'py>,
    strategies: Vec<PyReadonlyArray1<'py, f64>>,
    splits: i64,
) -> PyResult<Dict<'py>> {
    const OP: &str = "raptorbt.stats.pbo_cscv";
    let splits =
        count(OP, splits, &format!("splits must be an even number, at least {}", pbo::MIN_SPLITS))?;
    let cols = columns(&strategies);
    let r = py.detach(|| pbo::pbo_cscv(&cols, splits)).map_err(|e| fail(OP, e))?;
    let d = PyDict::new(py);
    d.set_item("n_periods", r.n_periods)?;
    d.set_item("n_strategies", r.n_strategies)?;
    d.set_item("splits", r.splits)?;
    d.set_item("combinations", r.combinations)?;
    d.set_item("dropped_rows", r.dropped_rows)?;
    d.set_item("pbo", r.pbo)?;
    d.set_item("median_logit", r.median_logit)?;
    d.set_item("mean_in_sample_sharpe_of_best", r.mean_in_sample_sharpe_of_best)?;
    d.set_item("mean_out_of_sample_sharpe_of_best", r.mean_out_of_sample_sharpe_of_best)?;
    d.set_item("prob_out_of_sample_loss_of_best", r.prob_out_of_sample_loss_of_best)?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (names, strategies, benchmark, reps, block_size, seed))]
fn stats_spa<'py>(
    py: Python<'py>,
    names: Vec<String>,
    strategies: Vec<PyReadonlyArray1<'py, f64>>,
    benchmark: Option<PyReadonlyArray1<'py, f64>>,
    reps: i64,
    block_size: Option<i64>,
    seed: u64,
) -> PyResult<Dict<'py>> {
    const OP: &str = "raptorbt.stats.spa_test";
    let message = format!("reps must be {} to {}", spa::MIN_REPS, spa::MAX_REPS);
    let reps = count(OP, reps, &message)?;
    let cols = columns(&strategies);
    let bench = benchmark.as_ref().map(numpy_cow_f64);
    let r = py
        .detach(|| spa::spa_test(&cols, bench.as_deref(), reps, block_size, seed))
        .map_err(|e| fail(OP, e))?;
    let boot = PyDict::new(py);
    boot.set_item("kind", "stationary")?;
    boot.set_item("mean_block", r.mean_block)?;
    boot.set_item("reps", r.reps)?;
    boot.set_item("seed", r.seed)?;
    let d = PyDict::new(py);
    d.set_item("n_periods", r.n_periods)?;
    d.set_item("n_strategies", r.n_strategies)?;
    d.set_item("best_strategy", &names[r.best_strategy])?;
    d.set_item("best_mean_excess", r.best_mean_excess)?;
    d.set_item("best_per_period_sharpe_excess", r.best_per_period_sharpe_excess)?;
    d.set_item("p_value_spa_consistent", r.p_value_spa_consistent)?;
    d.set_item("p_value_spa_lower", r.p_value_spa_lower)?;
    d.set_item("p_value_white_reality_check", r.p_value_white_reality_check)?;
    d.set_item("bootstrap", boot)?;
    Ok(d)
}

/// The p-values of given bootstrap resamples (arch's, in the oracle replay). `loss_diff` is one
/// array per strategy, the strategy minus the benchmark; `indices` is repetitions by periods.
#[pyfunction]
#[pyo3(signature = (loss_diff, block, indices))]
fn stats_spa_replay<'py>(
    py: Python<'py>,
    loss_diff: Vec<PyReadonlyArray1<'py, f64>>,
    block: usize,
    indices: PyReadonlyArray2<'py, i64>,
) -> PyResult<Dict<'py>> {
    const OP: &str = "raptorbt.stats.spa_replay";
    let cols = columns(&loss_diff);
    let mut rows: Vec<Vec<u32>> = Vec::with_capacity(indices.as_array().nrows());
    for row in indices.as_array().rows() {
        let mut out = Vec::with_capacity(row.len());
        for v in row {
            out.push(
                u32::try_from(*v).map_err(|_| invalid(OP, "a bootstrap index is out of range"))?,
            );
        }
        rows.push(out);
    }
    let r = py.detach(|| spa::spa_replay(&cols, block, &rows)).map_err(|e| fail(OP, e))?;
    let p = PyDict::new(py);
    p.set_item("lower", r.pvalues.lower)?;
    p.set_item("consistent", r.pvalues.consistent)?;
    p.set_item("upper", r.pvalues.upper)?;
    let d = PyDict::new(py);
    d.set_item("pvalues", p)?;
    d.set_item("loss_diff_mean", &r.loss_diff_mean)?;
    d.set_item("loss_diff_var", &r.loss_diff_var)?;
    d.set_item("valid", &r.valid)?;
    d.set_item("max_simulated", r.max_simulated.iter().map(|m| m.to_vec()).collect::<Vec<_>>())?;
    Ok(d)
}

// ------------------------------------------------------------------------------------ factors

#[pyfunction]
#[pyo3(signature = (names, returns, components, target))]
fn stats_pca<'py>(
    py: Python<'py>,
    names: Vec<String>,
    returns: Vec<PyReadonlyArray1<'py, f64>>,
    components: i64,
    target: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Dict<'py>> {
    const OP: &str = "raptorbt.stats.pca_factors";
    let components = count(OP, components, "components must be a positive number")?;
    let cols = columns(&returns);
    let tgt = target.as_ref().map(numpy_cow_f64);
    let r = py
        .detach(|| pca::pca_factors(&cols, components, tgt.as_deref()))
        .map_err(|e| fail(OP, e))?;
    let loadings = PyDict::new(py);
    for (j, row) in r.loadings.iter().enumerate().take(r.components) {
        loadings.set_item(format!("pc{}", j + 1), named(py, &names, row)?)?;
    }
    let d = PyDict::new(py);
    d.set_item("n", r.n)?;
    d.set_item("series", &names)?;
    d.set_item("components", r.components)?;
    d.set_item("explained_variance_ratio", &r.explained_variance_ratio)?;
    d.set_item("cumulative_explained", r.cumulative_explained)?;
    d.set_item("loadings", loadings)?;
    d.set_item("first_component_is_common_factor", r.first_component_is_common_factor)?;
    if let Some(t) = &r.target {
        let exposures = PyDict::new(py);
        for (j, e) in t.exposures_per_unit_score.iter().enumerate() {
            let cell = PyDict::new(py);
            cell.set_item("beta", e.beta)?;
            cell.set_item("t", e.t)?;
            exposures.set_item(format!("pc{}", j + 1), cell)?;
        }
        let target = PyDict::new(py);
        target.set_item("r_squared", t.r_squared)?;
        target.set_item("alpha_per_period", t.alpha_per_period)?;
        target.set_item("alpha_t", t.alpha_t)?;
        target.set_item("exposures_per_unit_score", exposures)?;
        d.set_item("target", target)?;
    }
    Ok(d)
}

// ------------------------------------------------------------------------------------ events

/// `dates` and `events` are day numbers (days since 1970-01-01).
#[pyfunction]
#[pyo3(signature = (dates, asset, market, events, estimation, gap, pre, post))]
#[allow(clippy::too_many_arguments)]
fn stats_events<'py>(
    py: Python<'py>,
    dates: Vec<i64>,
    asset: PyReadonlyArray1<'py, f64>,
    market: PyReadonlyArray1<'py, f64>,
    events: Vec<i64>,
    estimation: i64,
    gap: i64,
    pre: i64,
    post: i64,
) -> PyResult<Dict<'py>> {
    const OP: &str = "raptorbt.stats.event_study";
    let periods_msg =
        format!("estimation is at least {} periods, gap not negative", events::MIN_ESTIMATION);
    let window_msg = format!("pre and post are 0 to {} periods", events::MAX_WINDOW);
    let params = EventParams {
        estimation: count(OP, estimation, &periods_msg)?,
        gap: count(OP, gap, &periods_msg)?,
        pre: count(OP, pre, &window_msg)?,
        post: count(OP, post, &window_msg)?,
    };
    let (r, m) = (numpy_cow_f64(&asset), numpy_cow_f64(&market));
    let s = py
        .detach(|| events::event_study(&dates, &r, &m, &events, params))
        .map_err(|e| fail(OP, e))?;
    let rows = PyList::empty(py);
    for e in &s.events {
        let row = PyDict::new(py);
        row.set_item("event", iso_date(e.event))?;
        row.set_item("event_bar", iso_date(e.event_bar))?;
        row.set_item("alpha", e.alpha)?;
        row.set_item("beta", e.beta)?;
        row.set_item("residual_sigma", e.residual_sigma)?;
        row.set_item("abnormal_returns", &e.abnormal_returns)?;
        row.set_item("car", e.car)?;
        row.set_item("t", e.t)?;
        row.set_item("degenerate_fit", e.degenerate_fit)?;
        rows.append(row)?;
    }
    let skipped = PyList::empty(py);
    for k in &s.skipped {
        let row = PyDict::new(py);
        row.set_item("event", iso_date(k.event))?;
        row.set_item("reason", k.reason)?;
        skipped.append(row)?;
    }
    let window = PyDict::new(py);
    window.set_item("from", s.event_window.from)?;
    window.set_item("to", s.event_window.to)?;
    window.set_item("periods", s.event_window.periods)?;
    let estimation = PyDict::new(py);
    estimation.set_item("periods", s.estimation.periods)?;
    estimation.set_item("gap", s.estimation.gap)?;
    let d = PyDict::new(py);
    d.set_item("n_bars", s.n_bars)?;
    d.set_item("n_events", s.n_events)?;
    d.set_item("event_window", window)?;
    d.set_item("estimation", estimation)?;
    d.set_item("caar", s.caar)?;
    d.set_item("events", rows)?;
    d.set_item("skipped", skipped)?;
    d.set_item("caar_t", s.caar_t)?;
    if s.caar_t.is_some() {
        d.set_item("caar_p_value", s.caar_p_value)?;
        d.set_item("positive_share", s.positive_share)?;
    }
    Ok(d)
}

// -------------------------------------------------------------------------------------- cost

#[pyfunction]
#[pyo3(signature = (high, low, close, volume, quantity, spread_model, coefficient))]
#[allow(clippy::too_many_arguments)]
fn stats_impact<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    quantity: f64,
    spread_model: &str,
    coefficient: f64,
) -> PyResult<Dict<'py>> {
    const OP: &str = "raptorbt.stats.impact_estimate";
    let model = SpreadModel::parse(spread_model).map_err(|e| fail(OP, e))?;
    let (h, l, c, v) =
        (numpy_cow_f64(&high), numpy_cow_f64(&low), numpy_cow_f64(&close), numpy_cow_f64(&volume));
    let r = py
        .detach(|| impact::impact_estimate(&h, &l, &c, &v, quantity, model, coefficient))
        .map_err(|e| fail(OP, e))?;
    let i = &r.inputs;
    let inputs = PyDict::new(py);
    inputs.set_item("quantity", i.quantity)?;
    inputs.set_item("average_daily_volume", i.average_daily_volume)?;
    inputs.set_item("participation", i.participation)?;
    inputs.set_item("daily_sigma", i.daily_sigma)?;
    inputs.set_item("last_close", i.last_close)?;
    inputs.set_item("spread_model", i.spread_model.as_str())?;
    inputs.set_item("estimated_spread_bps", i.estimated_spread_bps)?;
    inputs.set_item("impact_coefficient", i.impact_coefficient)?;
    let d = PyDict::new(py);
    d.set_item("days", r.days)?;
    d.set_item("inputs", inputs)?;
    d.set_item("half_spread_bps", r.half_spread_bps)?;
    d.set_item("impact_bps", r.impact_bps)?;
    d.set_item("one_way_cost_bps", r.one_way_cost_bps)?;
    d.set_item("one_way_cost_per_unit", r.one_way_cost_per_unit)?;
    d.set_item("one_way_cost_total", r.one_way_cost_total)?;
    d.set_item("within_model_range", r.within_model_range)?;
    Ok(d)
}

// ------------------------------------------------------------------------------------ surface

#[pyfunction]
#[pyo3(signature = (names, cells))]
fn stats_surface<'py>(
    py: Python<'py>,
    names: Vec<String>,
    cells: Vec<(Vec<f64>, f64)>,
) -> PyResult<Dict<'py>> {
    const OP: &str = "raptorbt.stats.sensitivity_surface";
    let s = py.detach(|| surface::sensitivity_surface(&names, &cells)).map_err(|e| fail(OP, e))?;
    let axes = PyDict::new(py);
    for (n, a) in names.iter().zip(&s.axes) {
        axes.set_item(n, a)?;
    }
    let best = PyDict::new(py);
    best.set_item("params", named(py, &names, &s.best.params)?)?;
    best.set_item("metric", s.best.metric)?;
    best.set_item("neighbour_mean", s.best.neighbour_mean)?;
    best.set_item("neighbours", s.best.neighbours)?;
    best.set_item("plateau_share", s.best.plateau_share)?;
    let robust = PyDict::new(py);
    robust.set_item("params", named(py, &names, &s.robust_pick.params)?)?;
    robust.set_item("metric", s.robust_pick.metric)?;
    robust.set_item("neighbour_mean", s.robust_pick.neighbour_mean)?;
    let rows = PyList::empty(py);
    for c in &s.cells {
        let row = PyDict::new(py);
        row.set_item("params", named(py, &names, &c.params)?)?;
        row.set_item("metric", c.metric)?;
        row.set_item("neighbour_mean", c.neighbour_mean)?;
        row.set_item("neighbours", c.neighbours)?;
        row.set_item("lift_over_neighbours", c.lift_over_neighbours)?;
        rows.append(row)?;
    }
    let d = PyDict::new(py);
    d.set_item("parameters", &names)?;
    d.set_item("axes", axes)?;
    d.set_item("n_cells", s.n_cells)?;
    d.set_item("grid_cells", s.grid_cells)?;
    d.set_item("coverage", s.coverage)?;
    d.set_item("metric_min", s.metric_min)?;
    d.set_item("metric_max", s.metric_max)?;
    d.set_item("smoothness", s.smoothness)?;
    d.set_item("best", best)?;
    d.set_item("robust_pick", robust)?;
    d.set_item("verdict", s.verdict.as_str())?;
    d.set_item("cells", rows)?;
    if let Some(m) = &s.matrix {
        let matrix = PyDict::new(py);
        matrix.set_item("rows", &names[0])?;
        matrix.set_item("columns", &names[1])?;
        matrix.set_item("values", &m.values)?;
        d.set_item("matrix", matrix)?;
    }
    Ok(d)
}

// ------------------------------------------------------------------------------- data quality

#[pyfunction]
#[pyo3(signature = (
    t, open, high, low, close, volume, interval_seconds, outlier_z, mad_scale, listed,
    daily_gap_days, intraday_gap_bars, split_factors, split_tolerance, min_bars,
    dead_bar_share_max
))]
#[allow(clippy::too_many_arguments)]
fn stats_audit_bars<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    interval_seconds: u64,
    outlier_z: f64,
    mad_scale: f64,
    listed: usize,
    daily_gap_days: f64,
    intraday_gap_bars: f64,
    split_factors: Vec<f64>,
    split_tolerance: f64,
    min_bars: usize,
    dead_bar_share_max: f64,
) -> PyResult<Dict<'py>> {
    const OP: &str = "raptorbt.stats.audit_bars";
    let rules = AuditRules {
        outlier_z,
        mad_scale,
        listed,
        daily_gap_days,
        intraday_gap_bars,
        split_factors,
        split_tolerance,
        min_bars,
        dead_bar_share_max,
    };
    let (t, o, h, l, c, v) = (
        numpy_cow_f64(&t),
        numpy_cow_f64(&open),
        numpy_cow_f64(&high),
        numpy_cow_f64(&low),
        numpy_cow_f64(&close),
        numpy_cow_f64(&volume),
    );
    let a = py
        .detach(|| {
            let bars = Bars { t: &t, open: &o, high: &h, low: &l, close: &c, volume: &v };
            quality::audit_bars(&bars, interval_seconds, &rules)
        })
        .map_err(|e| fail(OP, e))?;
    let f = &a.findings;
    let impossible = PyDict::new(py);
    impossible.set_item("count", f.impossible_ohlc.count)?;
    impossible.set_item("first", &f.impossible_ohlc.first)?;
    let worst = PyList::empty(py);
    for o in &f.return_outliers {
        let row = PyDict::new(py);
        row.set_item("at", &o.at)?;
        row.set_item("return_pct", o.return_pct)?;
        worst.append(row)?;
    }
    let outliers = PyDict::new(py);
    outliers.set_item("robust_z_over", f.robust_z_over)?;
    outliers.set_item("worst", worst)?;
    let splits = PyList::empty(py);
    for s in &f.possible_unadjusted_splits {
        let row = PyDict::new(py);
        row.set_item("at", &s.at)?;
        row.set_item("close_ratio", s.close_ratio)?;
        splits.append(row)?;
    }
    let gaps = PyList::empty(py);
    for g in &f.gaps {
        let row = PyDict::new(py);
        match g {
            Gap::Daily { from, to, days } => {
                row.set_item("from", from)?;
                row.set_item("to", to)?;
                row.set_item("days", *days)?;
            }
            Gap::Intraday { from, to, missing_bars } => {
                row.set_item("from", from)?;
                row.set_item("to", to)?;
                row.set_item("missing_bars", *missing_bars)?;
            }
        }
        gaps.append(row)?;
    }
    let findings = PyDict::new(py);
    findings.set_item("duplicate_timestamps", f.duplicate_timestamps)?;
    findings.set_item("out_of_order", f.out_of_order)?;
    findings.set_item("impossible_ohlc", impossible)?;
    findings.set_item("zero_volume_bars", f.zero_volume_bars)?;
    findings.set_item("flat_bars", f.flat_bars)?;
    findings.set_item("flat_zero_volume_bars", f.flat_zero_volume_bars)?;
    findings.set_item("return_outliers", outliers)?;
    findings.set_item("possible_unadjusted_splits", splits)?;
    findings.set_item("gaps", gaps)?;
    let d = PyDict::new(py);
    d.set_item("bars", a.bars)?;
    d.set_item("first", &a.first)?;
    d.set_item("last", &a.last)?;
    d.set_item("findings", findings)?;
    d.set_item("clean", a.clean)?;
    Ok(d)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(stats_stationarity, m)?)?;
    m.add_function(wrap_pyfunction!(stats_cointegration, m)?)?;
    m.add_function(wrap_pyfunction!(stats_hurst, m)?)?;
    m.add_function(wrap_pyfunction!(stats_garch, m)?)?;
    m.add_function(wrap_pyfunction!(stats_regimes, m)?)?;
    m.add_function(wrap_pyfunction!(stats_hac, m)?)?;
    m.add_function(wrap_pyfunction!(stats_pbo, m)?)?;
    m.add_function(wrap_pyfunction!(stats_spa, m)?)?;
    m.add_function(wrap_pyfunction!(stats_spa_replay, m)?)?;
    m.add_function(wrap_pyfunction!(stats_pca, m)?)?;
    m.add_function(wrap_pyfunction!(stats_events, m)?)?;
    m.add_function(wrap_pyfunction!(stats_impact, m)?)?;
    m.add_function(wrap_pyfunction!(stats_surface, m)?)?;
    m.add_function(wrap_pyfunction!(stats_audit_bars, m)?)?;
    Ok(())
}
