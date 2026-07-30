//! Python bindings for the portfolio math surface.
//!
//! Thin conversion layer only: every computation lives in
//! `crate::portfolio::{covariance, optimize, factor_panel, risk_contrib,
//! rebalance}`. Errors map to `ValueError` with the Rust message verbatim,
//! so a Python caller sees exactly what was refused and why.

use std::collections::HashMap;

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::execution::indian_costs::{Segment, DP_SELL_CHARGE_PER_ISIN_PER_DAY};
use crate::portfolio::covariance::{ledoit_wolf, RiskModel};
use crate::portfolio::errors::PortfolioMathError;
use crate::portfolio::factor_panel;
use crate::portfolio::optimize::{optimize_long_only, OptimizationResult, OptimizerConfig};
use crate::portfolio::rebalance::{
    simulate_rebalance_policy as simulate_rebalance_policy_rs, RebalanceConfig, RebalancePolicy,
};
use crate::portfolio::risk_contrib::risk_contributions as risk_contributions_rs;

use super::numpy_bridge::{numpy_to_vec2_f64, numpy_to_vec_f64};

fn to_py_err(e: PortfolioMathError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn to_pyarray2<'py>(
    py: Python<'py>,
    flat: Vec<f64>,
    rows: usize,
    cols: usize,
) -> PyResult<&'py PyArray2<f64>> {
    Array2::from_shape_vec((rows, cols), flat)
        .map(|a| a.into_pyarray(py))
        .map_err(|e| PyValueError::new_err(format!("shape error: {e}")))
}

/// Parse a cost-segment name for the portfolio surface.
fn parse_segment(segment: &str) -> PyResult<Segment> {
    match segment.to_ascii_lowercase().as_str() {
        "equity_delivery" => Ok(Segment::EquityDelivery),
        "equity_intraday" => Ok(Segment::EquityIntraday),
        "futures_nfo" => Ok(Segment::FuturesNfo),
        "options_nfo" => Ok(Segment::OptionsNfo),
        "futures_mcx" => Ok(Segment::FuturesMcx),
        "options_mcx" => Ok(Segment::OptionsMcx),
        "futures_cds" => Ok(Segment::FuturesCds),
        "options_cds" => Ok(Segment::OptionsCds),
        other => Err(PyValueError::new_err(format!(
            "unknown segment '{other}'; expected one of equity_delivery, equity_intraday, \
             futures_nfo, options_nfo, futures_mcx, options_mcx, futures_cds, options_cds"
        ))),
    }
}

/// Shrunk covariance matrix plus the context needed to use it safely.
///
/// Carries `periods_per_year` and `asset_ids` so annualization and asset
/// ordering travel with the matrix; consumers validate against them instead
/// of trusting the caller.
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyRiskModel {
    pub(crate) inner: RiskModel,
}

#[pymethods]
impl PyRiskModel {
    /// Covariance of per-period returns as an `n_assets x n_assets` array.
    fn cov<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray2<f64>> {
        to_pyarray2(py, self.inner.cov.clone(), self.inner.n_assets, self.inner.n_assets)
    }

    #[getter]
    fn asset_ids(&self) -> Vec<String> {
        self.inner.asset_ids.clone()
    }

    #[getter]
    fn n_assets(&self) -> usize {
        self.inner.n_assets
    }

    #[getter]
    fn periods_per_year(&self) -> f64 {
        self.inner.periods_per_year
    }

    #[getter]
    fn shrinkage_intensity(&self) -> f64 {
        self.inner.shrinkage_intensity
    }

    #[getter]
    fn n_obs(&self) -> usize {
        self.inner.n_obs
    }

    fn __repr__(&self) -> String {
        format!(
            "PyRiskModel(n_assets={}, n_obs={}, shrinkage={:.4}, periods_per_year={})",
            self.inner.n_assets,
            self.inner.n_obs,
            self.inner.shrinkage_intensity,
            self.inner.periods_per_year
        )
    }
}

/// Constraint and objective configuration for the long-only optimizer.
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyOptimizerConfig {
    #[pyo3(get, set)]
    pub risk_aversion: f64,
    #[pyo3(get, set)]
    pub turnover_penalty: f64,
    #[pyo3(get, set)]
    pub position_cap: f64,
    #[pyo3(get, set)]
    pub sector_ids: Vec<usize>,
    #[pyo3(get, set)]
    pub sector_caps: Vec<f64>,
    #[pyo3(get, set)]
    pub no_trade_band: f64,
    #[pyo3(get, set)]
    pub min_trade_value: f64,
    #[pyo3(get, set)]
    pub portfolio_value: f64,
    #[pyo3(get, set)]
    pub cash_max: f64,
    #[pyo3(get, set)]
    pub max_iter: u32,
    #[pyo3(get, set)]
    pub tolerance: f64,
}

#[pymethods]
impl PyOptimizerConfig {
    #[new]
    #[pyo3(signature = (
        risk_aversion,
        turnover_penalty,
        position_cap,
        sector_ids,
        sector_caps,
        no_trade_band=0.0,
        min_trade_value=0.0,
        portfolio_value=0.0,
        cash_max=0.0,
        max_iter=200,
        tolerance=1e-8,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        risk_aversion: f64,
        turnover_penalty: f64,
        position_cap: f64,
        sector_ids: Vec<usize>,
        sector_caps: Vec<f64>,
        no_trade_band: f64,
        min_trade_value: f64,
        portfolio_value: f64,
        cash_max: f64,
        max_iter: u32,
        tolerance: f64,
    ) -> Self {
        Self {
            risk_aversion,
            turnover_penalty,
            position_cap,
            sector_ids,
            sector_caps,
            no_trade_band,
            min_trade_value,
            portfolio_value,
            cash_max,
            max_iter,
            tolerance,
        }
    }
}

impl PyOptimizerConfig {
    fn to_rust(&self) -> OptimizerConfig {
        OptimizerConfig {
            risk_aversion: self.risk_aversion,
            turnover_penalty: self.turnover_penalty,
            position_cap: self.position_cap,
            sector_ids: self.sector_ids.clone(),
            sector_caps: self.sector_caps.clone(),
            no_trade_band: self.no_trade_band,
            min_trade_value: self.min_trade_value,
            portfolio_value: self.portfolio_value,
            cash_max: self.cash_max,
            max_iter: self.max_iter,
            tolerance: self.tolerance,
        }
    }
}

/// Result of one optimization.
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyOptimizationResult {
    inner: OptimizationResult,
}

#[pymethods]
impl PyOptimizationResult {
    fn weights<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        PyArray1::from_vec(py, self.inner.weights.clone())
    }

    fn trades<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        PyArray1::from_vec(py, self.inner.trades.clone())
    }

    #[getter]
    fn snapped(&self) -> Vec<bool> {
        self.inner.snapped.clone()
    }

    #[getter]
    fn cash(&self) -> f64 {
        self.inner.cash
    }

    #[getter]
    fn turnover(&self) -> f64 {
        self.inner.turnover
    }

    #[getter]
    fn objective(&self) -> f64 {
        self.inner.objective
    }

    #[getter]
    fn vol_annualized(&self) -> f64 {
        self.inner.vol_annualized
    }

    #[getter]
    fn solver_status(&self) -> String {
        self.inner.solver_status.clone()
    }

    #[getter]
    fn iterations(&self) -> u32 {
        self.inner.iterations
    }

    fn __repr__(&self) -> String {
        format!(
            "PyOptimizationResult(turnover={:.4}, cash={:.4}, vol_annualized={:.4}, status={})",
            self.inner.turnover,
            self.inner.cash,
            self.inner.vol_annualized,
            self.inner.solver_status
        )
    }
}

/// Euler decomposition of portfolio volatility.
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyRiskContributions {
    total_vol_annualized: f64,
    marginal: Vec<f64>,
    contribution: Vec<f64>,
    pct_contribution: Vec<f64>,
}

#[pymethods]
impl PyRiskContributions {
    #[getter]
    fn total_vol_annualized(&self) -> f64 {
        self.total_vol_annualized
    }

    fn marginal<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        PyArray1::from_vec(py, self.marginal.clone())
    }

    fn contribution<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        PyArray1::from_vec(py, self.contribution.clone())
    }

    fn pct_contribution<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        PyArray1::from_vec(py, self.pct_contribution.clone())
    }
}

/// One user's book for `batch_optimize_portfolios`.
///
/// Eagerly copies arrays under the GIL at construction so the item is `Send`
/// and the batch loop can release the GIL (same pattern as
/// `PyBatchSpreadItem`).
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyOptimizeItem {
    item_id: String,
    alpha: Vec<f64>,
    w_current: Vec<f64>,
    portfolio_value: Option<f64>,
}

#[pymethods]
impl PyOptimizeItem {
    #[new]
    #[pyo3(signature = (item_id, alpha, w_current, portfolio_value=None))]
    fn new(
        item_id: String,
        alpha: PyReadonlyArray1<f64>,
        w_current: PyReadonlyArray1<f64>,
        portfolio_value: Option<f64>,
    ) -> Self {
        Self {
            item_id,
            alpha: numpy_to_vec_f64(alpha),
            w_current: numpy_to_vec_f64(w_current),
            portfolio_value,
        }
    }
}

/// Estimate a Ledoit-Wolf constant-correlation shrunk covariance.
///
/// `returns` is `n_obs x n_assets` per-period simple returns.
/// `periods_per_year` is required -- annualization is never inferred.
#[pyfunction]
pub fn estimate_covariance(
    returns: PyReadonlyArray2<f64>,
    asset_ids: Vec<String>,
    periods_per_year: f64,
) -> PyResult<PyRiskModel> {
    let (flat, n_obs, n_assets) = numpy_to_vec2_f64(returns);
    ledoit_wolf(&flat, n_obs, n_assets, asset_ids, periods_per_year)
        .map(|inner| PyRiskModel { inner })
        .map_err(to_py_err)
}

/// Optimize one long-only book seeded from current holdings.
///
/// `asset_ids` must match the model's ordering exactly -- a reordered
/// universe is an error, not a silently wrong result.
#[pyfunction]
pub fn optimize_portfolio(
    model: &PyRiskModel,
    alpha: PyReadonlyArray1<f64>,
    w_current: PyReadonlyArray1<f64>,
    asset_ids: Vec<String>,
    config: &PyOptimizerConfig,
) -> PyResult<PyOptimizationResult> {
    model.inner.require_same_assets(&asset_ids).map_err(to_py_err)?;
    let alpha = numpy_to_vec_f64(alpha);
    let w_current = numpy_to_vec_f64(w_current);
    optimize_long_only(&model.inner, &alpha, &w_current, &config.to_rust())
        .map(|inner| PyOptimizationResult { inner })
        .map_err(to_py_err)
}

/// Optimize many books against one risk model in parallel (Rayon).
///
/// Deterministic: results are identical to a serial loop, order-preserving
/// by item. A per-item error aborts the whole batch with that item named --
/// a batch with one silently missing user is worse than a failed batch.
#[pyfunction]
pub fn batch_optimize_portfolios(
    py: Python<'_>,
    model: &PyRiskModel,
    items: Vec<PyOptimizeItem>,
    config: &PyOptimizerConfig,
) -> PyResult<Vec<(String, PyOptimizationResult)>> {
    use rayon::prelude::*;

    let base = config.to_rust();
    let inner_model = model.inner.clone();

    let results: Vec<(String, Result<OptimizationResult, PortfolioMathError>)> =
        py.allow_threads(|| {
            items
                .into_par_iter()
                .map(|item| {
                    let mut cfg = base.clone();
                    if let Some(pv) = item.portfolio_value {
                        cfg.portfolio_value = pv;
                    }
                    let r = optimize_long_only(&inner_model, &item.alpha, &item.w_current, &cfg);
                    (item.item_id, r)
                })
                .collect()
        });

    results
        .into_iter()
        .map(|(id, r)| match r {
            Ok(inner) => Ok((id, PyOptimizationResult { inner })),
            Err(e) => Err(PyValueError::new_err(format!("item '{id}': {e}"))),
        })
        .collect()
}

/// Risk contributions of `weights` under `model`.
#[pyfunction]
pub fn compute_risk_contributions(
    model: &PyRiskModel,
    weights: PyReadonlyArray1<f64>,
    asset_ids: Vec<String>,
) -> PyResult<PyRiskContributions> {
    model.inner.require_same_assets(&asset_ids).map_err(to_py_err)?;
    let w = numpy_to_vec_f64(weights);
    risk_contributions_rs(&model.inner, &w)
        .map(|rc| PyRiskContributions {
            total_vol_annualized: rc.total_vol_annualized,
            marginal: rc.marginal,
            contribution: rc.contribution,
            pct_contribution: rc.pct_contribution,
        })
        .map_err(to_py_err)
}

/// Winsorize each date's cross-section (rows = dates, cols = assets).
#[pyfunction]
pub fn winsorize_panel<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<f64>,
    pct: f64,
) -> PyResult<&'py PyArray2<f64>> {
    let (flat, rows, cols) = numpy_to_vec2_f64(values);
    let out = factor_panel::winsorize_panel(&flat, rows, cols, pct).map_err(to_py_err)?;
    to_pyarray2(py, out, rows, cols)
}

/// Z-score each date's cross-section.
#[pyfunction]
pub fn zscore_panel<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<f64>,
    min_names: usize,
) -> PyResult<&'py PyArray2<f64>> {
    let (flat, rows, cols) = numpy_to_vec2_f64(values);
    let out = factor_panel::zscore_panel(&flat, rows, cols, min_names).map_err(to_py_err)?;
    to_pyarray2(py, out, rows, cols)
}

/// Rank each date's cross-section into [0, 1], ties averaged.
#[pyfunction]
pub fn rank_panel<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<f64>,
    min_names: usize,
) -> PyResult<&'py PyArray2<f64>> {
    let (flat, rows, cols) = numpy_to_vec2_f64(values);
    let out = factor_panel::rank_panel(&flat, rows, cols, min_names).map_err(to_py_err)?;
    to_pyarray2(py, out, rows, cols)
}

/// Price momentum with a skip window (12-1 daily: lookback=252, skip=21).
#[pyfunction]
pub fn momentum_panel<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray2<f64>,
    lookback: usize,
    skip: usize,
) -> PyResult<&'py PyArray2<f64>> {
    let (flat, rows, cols) = numpy_to_vec2_f64(prices);
    let out = factor_panel::momentum_panel(&flat, rows, cols, lookback, skip).map_err(to_py_err)?;
    to_pyarray2(py, out, rows, cols)
}

/// Weighted composite of same-shaped factor panels.
#[pyfunction]
pub fn composite_scores<'py>(
    py: Python<'py>,
    factors: Vec<PyReadonlyArray2<f64>>,
    weights: PyReadonlyArray1<f64>,
) -> PyResult<&'py PyArray2<f64>> {
    if factors.is_empty() {
        return Err(PyValueError::new_err("no factor panels supplied"));
    }
    let converted: Vec<(Vec<f64>, usize, usize)> =
        factors.into_iter().map(numpy_to_vec2_f64).collect();
    let (rows, cols) = (converted[0].1, converted[0].2);
    for (i, (_, r, c)) in converted.iter().enumerate() {
        if *r != rows || *c != cols {
            return Err(PyValueError::new_err(format!(
                "factor {i} shape {r}x{c} differs from factor 0 shape {rows}x{cols}"
            )));
        }
    }
    let slices: Vec<&[f64]> = converted.iter().map(|(v, _, _)| v.as_slice()).collect();
    let w = numpy_to_vec_f64(weights);
    let out = factor_panel::composite_scores(&slices, &w, rows, cols).map_err(to_py_err)?;
    to_pyarray2(py, out, rows, cols)
}

/// Measured rank IC of a factor against forward returns.
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyRankIc {
    mean_ic: f64,
    stdev_ic: f64,
    t_stat: f64,
    n_dates_scored: usize,
    mean_names: f64,
    daily_ic: Vec<f64>,
}

#[pymethods]
impl PyRankIc {
    #[getter]
    fn mean_ic(&self) -> f64 {
        self.mean_ic
    }

    #[getter]
    fn stdev_ic(&self) -> f64 {
        self.stdev_ic
    }

    #[getter]
    fn t_stat(&self) -> f64 {
        self.t_stat
    }

    #[getter]
    fn n_dates_scored(&self) -> usize {
        self.n_dates_scored
    }

    #[getter]
    fn mean_names(&self) -> f64 {
        self.mean_names
    }

    fn daily_ic<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        PyArray1::from_vec(py, self.daily_ic.clone())
    }
}

/// Rank IC of a factor panel against `horizon`-ahead returns from `prices`.
///
/// Forward returns are derived here rather than supplied, so the shift cannot
/// leak lookahead. Dates with fewer than `min_names` paired observations are
/// skipped, never zero-filled.
#[pyfunction]
pub fn rank_ic(
    factor: PyReadonlyArray2<f64>,
    prices: PyReadonlyArray2<f64>,
    horizon: usize,
    min_names: usize,
) -> PyResult<PyRankIc> {
    let (f_flat, rows, cols) = numpy_to_vec2_f64(factor);
    let (p_flat, p_rows, p_cols) = numpy_to_vec2_f64(prices);
    if rows != p_rows || cols != p_cols {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "factor is {rows}x{cols} but prices is {p_rows}x{p_cols}"
        )));
    }
    let out = factor_panel::rank_ic(&f_flat, &p_flat, rows, cols, horizon, min_names)
        .map_err(to_py_err)?;
    Ok(PyRankIc {
        mean_ic: out.mean_ic,
        stdev_ic: out.stdev_ic,
        t_stat: out.t_stat,
        n_dates_scored: out.n_dates_scored,
        mean_names: out.mean_names,
        daily_ic: out.daily_ic,
    })
}

/// Result of a rebalance-policy simulation.
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyRebalanceSimResult {
    equity_curve: Vec<f64>,
    turnover: Vec<f64>,
    cost_regulatory: Vec<f64>,
    cost_brokerage: Vec<f64>,
    cost_dp: Vec<f64>,
    n_rebalances: u32,
    n_trades: u32,
    total_cost_drag_annualized: f64,
}

#[pymethods]
impl PyRebalanceSimResult {
    fn equity_curve<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        PyArray1::from_vec(py, self.equity_curve.clone())
    }

    fn turnover<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        PyArray1::from_vec(py, self.turnover.clone())
    }

    fn cost_regulatory<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        PyArray1::from_vec(py, self.cost_regulatory.clone())
    }

    fn cost_brokerage<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        PyArray1::from_vec(py, self.cost_brokerage.clone())
    }

    fn cost_dp<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        PyArray1::from_vec(py, self.cost_dp.clone())
    }

    #[getter]
    fn n_rebalances(&self) -> u32 {
        self.n_rebalances
    }

    #[getter]
    fn n_trades(&self) -> u32 {
        self.n_trades
    }

    #[getter]
    fn total_cost_drag_annualized(&self) -> f64 {
        self.total_cost_drag_annualized
    }
}

/// Simulate following a target-weight series under a rebalance policy.
///
/// `policy` is "calendar" (with `policy_param` = every N dates, >= 1) or
/// "band" (with `policy_param` = one-way drift fraction). Costs use the
/// named segment's per-leg schedule plus `dp_charge_per_isin` once per
/// distinct asset with a net sell per rebalance date.
#[pyfunction]
#[pyo3(signature = (
    prices,
    target_weights,
    initial_capital,
    policy,
    policy_param,
    segment="equity_delivery",
    min_trade_value=0.0,
    dp_charge_per_isin=DP_SELL_CHARGE_PER_ISIN_PER_DAY,
    periods_per_year=252.0,
))]
#[allow(clippy::too_many_arguments)]
pub fn simulate_rebalance_policy(
    prices: PyReadonlyArray2<f64>,
    target_weights: PyReadonlyArray2<f64>,
    initial_capital: f64,
    policy: &str,
    policy_param: f64,
    segment: &str,
    min_trade_value: f64,
    dp_charge_per_isin: f64,
    periods_per_year: f64,
) -> PyResult<PyRebalanceSimResult> {
    let (px, n_dates, n_assets) = numpy_to_vec2_f64(prices);
    let (tw, tw_dates, tw_assets) = numpy_to_vec2_f64(target_weights);
    if tw_dates != n_dates || tw_assets != n_assets {
        return Err(PyValueError::new_err(format!(
            "target_weights shape {tw_dates}x{tw_assets} differs from prices {n_dates}x{n_assets}"
        )));
    }
    let policy = match policy.to_ascii_lowercase().as_str() {
        "calendar" => {
            if policy_param < 1.0 || policy_param.fract() != 0.0 {
                return Err(PyValueError::new_err(format!(
                    "calendar policy_param must be a whole number >= 1, got {policy_param}"
                )));
            }
            RebalancePolicy::Calendar { every_n: policy_param as usize }
        }
        "band" => RebalancePolicy::Band { band: policy_param },
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown policy '{other}'; expected 'calendar' or 'band'"
            )))
        }
    };
    let cfg = RebalanceConfig {
        initial_capital,
        policy,
        min_trade_value,
        segment: parse_segment(segment)?,
        dp_charge_per_isin,
        periods_per_year,
    };
    simulate_rebalance_policy_rs(&px, n_dates, n_assets, &tw, &cfg)
        .map(|r| PyRebalanceSimResult {
            equity_curve: r.equity_curve,
            turnover: r.turnover,
            cost_regulatory: r.cost_regulatory,
            cost_brokerage: r.cost_brokerage,
            cost_dp: r.cost_dp,
            n_rebalances: r.n_rebalances,
            n_trades: r.n_trades,
            total_cost_drag_annualized: r.total_cost_drag_annualized,
        })
        .map_err(to_py_err)
}

/// Export a segment's cost schedule, including the DP sell charge.
///
/// The single numeric source of truth for cost parity tests between this
/// crate and the backend's `costs.py` -- the two must never drift.
#[pyfunction]
pub fn indian_cost_schedule(segment: &str) -> PyResult<HashMap<String, f64>> {
    let seg = parse_segment(segment)?;
    let s = seg.schedule();
    let mut out = HashMap::new();
    out.insert("brokerage_per_order".into(), s.brokerage_per_order);
    out.insert("stt_rate".into(), s.stt_rate);
    out.insert("exchange_txn_rate".into(), s.exchange_txn_rate);
    out.insert("sebi_turnover_rate".into(), s.sebi_turnover_rate);
    out.insert("stamp_duty_rate".into(), s.stamp_duty_rate);
    out.insert("gst_rate".into(), s.gst_rate);
    out.insert("dp_sell_charge_per_isin_per_day".into(), DP_SELL_CHARGE_PER_ISIN_PER_DAY);
    Ok(out)
}
