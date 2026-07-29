//! Constrained long-only portfolio optimizer.
//!
//! Solves, over weights `w` (fraction of portfolio value per asset):
//!
//! ```text
//! maximize   alpha'w  -  risk_aversion * w' Sigma w  -  turnover_penalty * ||w - w_current||_1
//! subject to sum(w) + cash = 1,   0 <= cash <= cash_max
//!            0 <= w_i <= position_cap
//!            sum_{i in sector k} w_i <= sector_caps[k]
//! ```
//!
//! via the Clarabel interior-point solver (the L1 term is reformulated exactly
//! with auxiliary variables `t_i >= |w_i - w_current_i|`). Interior-point over
//! a hand-rolled projected gradient is deliberate: the feasible set
//! {simplex-with-cash ∩ box ∩ sector polytope} has no closed-form projection,
//! and Clarabel certifies infeasibility / non-convergence as a hard status --
//! which is what lets this function refuse instead of returning a plausible
//! but uncertified iterate.
//!
//! The no-trade band and minimum trade value are non-convex, so they are
//! applied *post-solve*: small diffs snap back to the current weight and the
//! residual goes to explicit cash -- never rescaled across other names, which
//! could breach a cap. If snapping strands cash outside [0, cash_max], the
//! function errors with the amounts rather than clamping.

use clarabel::algebra::CscMatrix;
use clarabel::solver::{
    DefaultSettingsBuilder, DefaultSolver, IPSolver, NonnegativeConeT, SolverStatus,
    SupportedConeT,
};

use super::covariance::RiskModel;
use super::errors::PortfolioMathError;

/// Constraint and objective configuration for one optimization.
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Lambda on the w'Sigma w term. Must be > 0.
    pub risk_aversion: f64,
    /// Gamma on the L1 turnover term. Must be >= 0.
    pub turnover_penalty: f64,
    /// Per-asset weight cap in (0, 1].
    pub position_cap: f64,
    /// Sector index per asset (0-based into `sector_caps`).
    pub sector_ids: Vec<usize>,
    /// Weight cap per sector.
    pub sector_caps: Vec<f64>,
    /// Post-solve: |delta_w| below this snaps to the current weight.
    pub no_trade_band: f64,
    /// Post-solve: trades below this rupee value snap to the current weight.
    pub min_trade_value: f64,
    /// Portfolio value in rupees (used only with `min_trade_value`).
    pub portfolio_value: f64,
    /// Maximum cash fraction in [0, 1).
    pub cash_max: f64,
    /// Solver iteration cap.
    pub max_iter: u32,
    /// Solver feasibility/gap tolerance.
    pub tolerance: f64,
}

/// Result of one optimization.
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Final weights after post-solve snapping.
    pub weights: Vec<f64>,
    /// Final weight changes (weights - w_current).
    pub trades: Vec<f64>,
    /// Which assets were snapped back to their current weight.
    pub snapped: Vec<bool>,
    /// Final cash fraction.
    pub cash: f64,
    /// One-way turnover: 0.5 * sum(|trades|).
    pub turnover: f64,
    /// Solver objective value (pre-snap, minimization form).
    pub objective: f64,
    /// Annualized volatility of the final weights under the model.
    pub vol_annualized: f64,
    /// Clarabel termination status, for the audit trail.
    pub solver_status: String,
    /// Interior-point iterations used.
    pub iterations: u32,
}

fn validate(
    model: &RiskModel,
    alpha: &[f64],
    w_current: &[f64],
    cfg: &OptimizerConfig,
) -> Result<(), PortfolioMathError> {
    let n = model.n_assets;
    if alpha.len() != n || w_current.len() != n {
        return Err(PortfolioMathError::ShapeMismatch(format!(
            "model has {n} assets; alpha has {}, w_current has {}",
            alpha.len(),
            w_current.len()
        )));
    }
    for (i, v) in alpha.iter().enumerate() {
        if !v.is_finite() {
            return Err(PortfolioMathError::NonFinite { row: 0, col: i });
        }
    }
    let mut w_sum = 0.0;
    for (i, v) in w_current.iter().enumerate() {
        if !v.is_finite() {
            return Err(PortfolioMathError::NonFinite { row: 1, col: i });
        }
        if *v < -1e-12 {
            return Err(PortfolioMathError::DegenerateInput(format!(
                "w_current[{i}] = {v} is negative; this optimizer is long-only"
            )));
        }
        w_sum += v;
    }
    if w_sum > 1.0 + 1e-6 {
        return Err(PortfolioMathError::DegenerateInput(format!(
            "w_current sums to {w_sum:.6}, which exceeds 1"
        )));
    }
    if !(cfg.risk_aversion.is_finite() && cfg.risk_aversion > 0.0) {
        return Err(PortfolioMathError::DegenerateInput(format!(
            "risk_aversion must be > 0, got {}",
            cfg.risk_aversion
        )));
    }
    if !(cfg.turnover_penalty.is_finite() && cfg.turnover_penalty >= 0.0) {
        return Err(PortfolioMathError::DegenerateInput(format!(
            "turnover_penalty must be >= 0, got {}",
            cfg.turnover_penalty
        )));
    }
    if !(cfg.position_cap > 0.0 && cfg.position_cap <= 1.0) {
        return Err(PortfolioMathError::DegenerateInput(format!(
            "position_cap must be in (0, 1], got {}",
            cfg.position_cap
        )));
    }
    if !(0.0..1.0).contains(&cfg.cash_max) {
        return Err(PortfolioMathError::DegenerateInput(format!(
            "cash_max must be in [0, 1), got {}",
            cfg.cash_max
        )));
    }
    if cfg.no_trade_band < 0.0 || cfg.min_trade_value < 0.0 {
        return Err(PortfolioMathError::DegenerateInput(
            "no_trade_band and min_trade_value must be >= 0".into(),
        ));
    }
    if cfg.min_trade_value > 0.0 && !(cfg.portfolio_value.is_finite() && cfg.portfolio_value > 0.0)
    {
        return Err(PortfolioMathError::DegenerateInput(format!(
            "min_trade_value set but portfolio_value is {}",
            cfg.portfolio_value
        )));
    }
    if cfg.sector_ids.len() != n {
        return Err(PortfolioMathError::ShapeMismatch(format!(
            "{} sector_ids for {n} assets",
            cfg.sector_ids.len()
        )));
    }
    let n_sectors = cfg.sector_caps.len();
    for (i, &s) in cfg.sector_ids.iter().enumerate() {
        if s >= n_sectors {
            return Err(PortfolioMathError::ShapeMismatch(format!(
                "sector_ids[{i}] = {s} out of range for {n_sectors} sector_caps"
            )));
        }
    }
    for (k, &c) in cfg.sector_caps.iter().enumerate() {
        if !(c.is_finite() && c > 0.0) {
            return Err(PortfolioMathError::DegenerateInput(format!(
                "sector_caps[{k}] must be > 0, got {c}"
            )));
        }
    }

    // Feasibility arithmetic before handing Clarabel an impossible problem:
    // the caps must admit at least (1 - cash_max) of investment.
    let required = 1.0 - cfg.cash_max;
    if cfg.position_cap * n as f64 + 1e-12 < required {
        return Err(PortfolioMathError::Infeasible(format!(
            "position_cap {} x {n} assets = {:.4} cannot reach required investment {:.4}",
            cfg.position_cap,
            cfg.position_cap * n as f64,
            required
        )));
    }
    let mut sector_counts = vec![0usize; n_sectors];
    for &s in &cfg.sector_ids {
        sector_counts[s] += 1;
    }
    let reachable: f64 = (0..n_sectors)
        .map(|k| cfg.sector_caps[k].min(cfg.position_cap * sector_counts[k] as f64))
        .sum();
    if reachable + 1e-12 < required {
        return Err(PortfolioMathError::Infeasible(format!(
            "sector caps admit at most {reachable:.4} invested, below required {required:.4}"
        )));
    }
    Ok(())
}

/// Optimize a long-only book seeded from current holdings.
pub fn optimize_long_only(
    model: &RiskModel,
    alpha: &[f64],
    w_current: &[f64],
    cfg: &OptimizerConfig,
) -> Result<OptimizationResult, PortfolioMathError> {
    validate(model, alpha, w_current, cfg)?;
    let n = model.n_assets;
    let n_sectors = cfg.sector_caps.len();
    let nv = 2 * n; // variables: [w; t]

    // P (upper triangle only): 2 * risk_aversion * Sigma on the w block.
    let mut p_i = Vec::new();
    let mut p_j = Vec::new();
    let mut p_v = Vec::new();
    for i in 0..n {
        for j in i..n {
            let v = 2.0 * cfg.risk_aversion * model.cov[i * n + j];
            if v != 0.0 {
                p_i.push(i);
                p_j.push(j);
                p_v.push(v);
            }
        }
    }
    let p = CscMatrix::new_from_triplets(nv, nv, p_i, p_j, p_v);

    // q: minimize -alpha'w + gamma * sum(t).
    let mut q = vec![0.0; nv];
    for i in 0..n {
        q[i] = -alpha[i];
        q[n + i] = cfg.turnover_penalty;
    }

    // Inequality rows (Ax <= b), all in the nonnegative cone.
    let m = 2 + n + n + n_sectors + 2 * n;
    let mut a_i = Vec::new();
    let mut a_j = Vec::new();
    let mut a_v = Vec::new();
    let mut b = Vec::with_capacity(m);
    let mut row = 0usize;

    // sum(w) <= 1
    for i in 0..n {
        a_i.push(row);
        a_j.push(i);
        a_v.push(1.0);
    }
    b.push(1.0);
    row += 1;
    // -sum(w) <= cash_max - 1  (i.e. sum(w) >= 1 - cash_max)
    for i in 0..n {
        a_i.push(row);
        a_j.push(i);
        a_v.push(-1.0);
    }
    b.push(cfg.cash_max - 1.0);
    row += 1;
    // -w_i <= 0
    for i in 0..n {
        a_i.push(row);
        a_j.push(i);
        a_v.push(-1.0);
        b.push(0.0);
        row += 1;
    }
    // w_i <= position_cap
    for i in 0..n {
        a_i.push(row);
        a_j.push(i);
        a_v.push(1.0);
        b.push(cfg.position_cap);
        row += 1;
    }
    // sector sums <= sector_caps
    for k in 0..n_sectors {
        for i in 0..n {
            if cfg.sector_ids[i] == k {
                a_i.push(row);
                a_j.push(i);
                a_v.push(1.0);
            }
        }
        b.push(cfg.sector_caps[k]);
        row += 1;
    }
    // w_i - t_i <= w_current_i   and   -w_i - t_i <= -w_current_i
    for i in 0..n {
        a_i.push(row);
        a_j.push(i);
        a_v.push(1.0);
        a_i.push(row);
        a_j.push(n + i);
        a_v.push(-1.0);
        b.push(w_current[i]);
        row += 1;
    }
    for i in 0..n {
        a_i.push(row);
        a_j.push(i);
        a_v.push(-1.0);
        a_i.push(row);
        a_j.push(n + i);
        a_v.push(-1.0);
        b.push(-w_current[i]);
        row += 1;
    }
    debug_assert_eq!(row, m);
    let a = CscMatrix::new_from_triplets(m, nv, a_i, a_j, a_v);
    let cones: Vec<SupportedConeT<f64>> = vec![NonnegativeConeT(m)];

    let settings = DefaultSettingsBuilder::default()
        .verbose(false)
        .max_iter(cfg.max_iter)
        .tol_gap_abs(cfg.tolerance)
        .tol_gap_rel(cfg.tolerance)
        .tol_feas(cfg.tolerance)
        .build()
        .map_err(|e| PortfolioMathError::SolverFailed(format!("settings: {e}")))?;

    let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, settings)
        .map_err(|e| PortfolioMathError::SolverFailed(format!("setup: {e:?}")))?;
    solver.solve();

    let status = solver.solution.status;
    match status {
        SolverStatus::Solved => {}
        SolverStatus::PrimalInfeasible | SolverStatus::AlmostPrimalInfeasible => {
            return Err(PortfolioMathError::Infeasible(format!(
                "solver certified primal infeasibility ({status:?})"
            )));
        }
        other => {
            return Err(PortfolioMathError::SolverFailed(format!(
                "status {other:?} after {} iterations",
                solver.info.iterations
            )));
        }
    }

    let mut weights: Vec<f64> = solver.solution.x[..n].to_vec();
    // Interior-point solutions sit strictly inside the cone; clean sub-tolerance
    // negatives introduced by the solver itself (not by input data).
    for w in weights.iter_mut() {
        if *w < 0.0 && *w > -1e-9 {
            *w = 0.0;
        }
    }

    // Post-solve non-convex filters: snap small diffs back to current.
    let mut snapped = vec![false; n];
    for i in 0..n {
        let delta = weights[i] - w_current[i];
        let below_band = delta.abs() < cfg.no_trade_band;
        let below_value = cfg.min_trade_value > 0.0
            && delta.abs() * cfg.portfolio_value < cfg.min_trade_value;
        if (below_band || below_value) && delta != 0.0 {
            weights[i] = w_current[i];
            snapped[i] = true;
        }
    }

    let invested: f64 = weights.iter().sum();
    let mut cash = 1.0 - invested;
    let eps = (10.0 * cfg.tolerance).max(1e-9);
    let no_trade = weights
        .iter()
        .zip(w_current.iter())
        .all(|(w, c)| (w - c).abs() < eps);
    if no_trade {
        // Every diff snapped away: the status-quo book stands. Its cash is
        // whatever it already is -- the cash_max bound governs PROPOSED
        // books, not the pre-existing one (a book that exists is feasible
        // by definition). "No trade worth making" is a result, not an error.
        weights.copy_from_slice(w_current);
        cash = (1.0 - w_current.iter().sum::<f64>()).clamp(0.0, 1.0);
    } else {
        // A PARTIAL snap that strands cash outside the bound is a genuinely
        // broken half-rebalance: refuse with the arithmetic. Sub-tolerance
        // overshoot is solver noise, not stranded weight.
        if cash < -eps || cash > cfg.cash_max + eps {
            return Err(PortfolioMathError::Infeasible(format!(
                "post-snap cash {cash:.6} outside [0, {:.6}]; snapping stranded \
                 {:.6} of weight -- widen cash_max, lower the band, or accept the trades",
                cfg.cash_max,
                if cash < 0.0 { cash } else { cash - cfg.cash_max }
            )));
        }
        cash = cash.clamp(0.0, cfg.cash_max);
    }

    let trades: Vec<f64> = weights
        .iter()
        .zip(w_current.iter())
        .map(|(w, c)| w - c)
        .collect();
    let turnover = 0.5 * trades.iter().map(|t| t.abs()).sum::<f64>();

    // Annualized vol of the final book.
    let mut variance = 0.0;
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..n {
            acc += model.cov[i * n + j] * weights[j];
        }
        variance += weights[i] * acc;
    }
    let vol_annualized = variance.max(0.0).sqrt() * model.periods_per_year.sqrt();

    Ok(OptimizationResult {
        weights,
        trades,
        snapped,
        cash,
        turnover,
        objective: solver.solution.obj_val,
        vol_annualized,
        solver_status: format!("{status:?}"),
        iterations: solver.info.iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(cov: Vec<f64>, n: usize) -> RiskModel {
        RiskModel {
            cov,
            n_assets: n,
            asset_ids: (0..n).map(|i| format!("A{i}")).collect(),
            periods_per_year: 252.0,
            shrinkage_intensity: 0.1,
            n_obs: 500,
        }
    }

    fn base_cfg(n: usize) -> OptimizerConfig {
        OptimizerConfig {
            risk_aversion: 1.0,
            turnover_penalty: 0.0,
            position_cap: 1.0,
            sector_ids: vec![0; n],
            sector_caps: vec![1.0],
            no_trade_band: 0.0,
            min_trade_value: 0.0,
            portfolio_value: 1_000_000.0,
            cash_max: 0.0,
            max_iter: 200,
            tolerance: 1e-9,
        }
    }

    #[test]
    fn two_asset_unconstrained_matches_closed_form() {
        // Equal alphas, uncorrelated assets, no caps binding, fully invested:
        // with sum(w)=1, minimizing w'Sigma w gives the inverse-variance split
        // w1 = s2/(s1+s2).
        let m = model(vec![0.04, 0.0, 0.0, 0.08], 2);
        let cfg = OptimizerConfig {
            risk_aversion: 10.0,
            ..base_cfg(2)
        };
        let r = optimize_long_only(&m, &[0.0, 0.0], &[0.5, 0.5], &cfg).unwrap();
        let expect_w0 = 0.08 / (0.04 + 0.08);
        assert!((r.weights[0] - expect_w0).abs() < 1e-4, "{:?}", r.weights);
        assert!((r.weights[0] + r.weights[1] - 1.0).abs() < 1e-6);
        assert!((r.cash).abs() < 1e-6);
    }

    #[test]
    fn position_cap_binds() {
        // Asset 0 has huge alpha; cap forces the excess into asset 1 and 2.
        let m = model(
            vec![0.04, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.04],
            3,
        );
        let mut cfg = base_cfg(3);
        cfg.position_cap = 0.4;
        let r = optimize_long_only(&m, &[10.0, 0.0, 0.0], &[1.0 / 3.0; 3], &cfg).unwrap();
        assert!((r.weights[0] - 0.4).abs() < 1e-5, "{:?}", r.weights);
    }

    #[test]
    fn sector_cap_binds() {
        // Assets 0,1 share sector 0 capped at 0.5; asset 2 alone in sector 1.
        let m = model(
            vec![0.04, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.04],
            3,
        );
        let mut cfg = base_cfg(3);
        cfg.sector_ids = vec![0, 0, 1];
        cfg.sector_caps = vec![0.5, 1.0];
        let r = optimize_long_only(&m, &[5.0, 5.0, 0.0], &[1.0 / 3.0; 3], &cfg).unwrap();
        assert!(r.weights[0] + r.weights[1] <= 0.5 + 1e-6, "{:?}", r.weights);
        assert!((r.weights[2] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn huge_turnover_penalty_freezes_book() {
        let m = model(vec![0.04, 0.01, 0.01, 0.08], 2);
        let mut cfg = base_cfg(2);
        cfg.turnover_penalty = 1e6;
        let w_cur = [0.7, 0.3];
        let r = optimize_long_only(&m, &[0.5, -0.5], &w_cur, &cfg).unwrap();
        assert!(r.turnover < 1e-6, "turnover {}", r.turnover);
        assert!((r.weights[0] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn turnover_monotone_in_penalty() {
        let m = model(vec![0.04, 0.01, 0.01, 0.08], 2);
        let w_cur = [0.9, 0.1];
        let alpha = [0.0, 0.5];
        let mut prev = f64::INFINITY;
        for gamma in [0.0, 0.01, 0.1, 1.0] {
            let mut cfg = base_cfg(2);
            cfg.turnover_penalty = gamma;
            let r = optimize_long_only(&m, &alpha, &w_cur, &cfg).unwrap();
            assert!(
                r.turnover <= prev + 1e-9,
                "turnover not monotone: {} then {} at gamma={gamma}",
                prev,
                r.turnover
            );
            prev = r.turnover;
        }
    }

    #[test]
    fn no_trade_band_snaps_small_diff() {
        // Fully invested (cash_max = 0) so the zero-alpha optimum is exactly
        // 50/50; current is 50.5/49.5 -- inside the band.
        let m = model(vec![0.04, 0.0, 0.0, 0.04], 2);
        let mut cfg = base_cfg(2);
        cfg.no_trade_band = 0.02;
        let r = optimize_long_only(&m, &[0.0, 0.0], &[0.505, 0.495], &cfg).unwrap();
        assert!(r.snapped[0] && r.snapped[1], "{:?}", r.snapped);
        assert_eq!(r.weights, vec![0.505, 0.495]);
        assert_eq!(r.turnover, 0.0);
    }

    #[test]
    fn all_trades_snapped_returns_status_quo_not_infeasible() {
        // A tiny book where every model buy is below min_trade_value: the
        // result is the CURRENT book (turnover 0), even though its cash
        // fraction exceeds cash_max -- the bound governs proposed books.
        let m = model(vec![0.04, 0.0, 0.0, 0.04], 2);
        let mut cfg = base_cfg(2);
        cfg.cash_max = 0.05;
        cfg.min_trade_value = 5_000.0;
        cfg.portfolio_value = 400.0; // every possible trade is < Rs 5,000
        let w_cur = [0.10, 0.05]; // 85% effectively uninvested
        let r = optimize_long_only(&m, &[0.5, 0.5], &w_cur, &cfg).unwrap();
        assert_eq!(r.weights, w_cur.to_vec());
        assert_eq!(r.turnover, 0.0);
        assert!((r.cash - 0.85).abs() < 1e-9);
    }

    #[test]
    fn min_trade_value_filters_small_trade() {
        let m = model(vec![0.04, 0.0, 0.0, 0.04], 2);
        let mut cfg = base_cfg(2);
        cfg.min_trade_value = 5_000.0;
        cfg.portfolio_value = 100_000.0; // 5% of book is the floor
        let r = optimize_long_only(&m, &[0.0, 0.0], &[0.52, 0.48], &cfg).unwrap();
        // The 2% rebalance trade is worth 2000 < 5000: snapped.
        assert!(r.snapped[0] && r.snapped[1]);
        assert_eq!(r.turnover, 0.0);
    }

    #[test]
    fn infeasible_caps_refused_before_solving() {
        let m = model(vec![0.04, 0.0, 0.0, 0.04], 2);
        let mut cfg = base_cfg(2);
        cfg.position_cap = 0.3; // 2 x 0.3 = 0.6 < 1.0 required
        let err = optimize_long_only(&m, &[0.0, 0.0], &[0.5, 0.5], &cfg).unwrap_err();
        assert!(matches!(err, PortfolioMathError::Infeasible(_)), "{err}");
    }

    #[test]
    fn deterministic_across_runs() {
        let m = model(vec![0.04, 0.01, 0.01, 0.08], 2);
        let cfg = OptimizerConfig {
            turnover_penalty: 0.05,
            ..base_cfg(2)
        };
        let a = optimize_long_only(&m, &[0.3, 0.1], &[0.6, 0.4], &cfg).unwrap();
        let b = optimize_long_only(&m, &[0.3, 0.1], &[0.6, 0.4], &cfg).unwrap();
        assert_eq!(a.weights, b.weights);
        assert_eq!(a.objective, b.objective);
        assert_eq!(a.iterations, b.iterations);
    }

    #[test]
    fn refuses_nan_alpha_and_overweight_current() {
        let m = model(vec![0.04, 0.0, 0.0, 0.04], 2);
        let cfg = base_cfg(2);
        assert!(optimize_long_only(&m, &[f64::NAN, 0.0], &[0.5, 0.5], &cfg).is_err());
        assert!(optimize_long_only(&m, &[0.0, 0.0], &[0.9, 0.9], &cfg).is_err());
        assert!(optimize_long_only(&m, &[0.0, 0.0], &[-0.1, 0.5], &cfg).is_err());
    }
}
