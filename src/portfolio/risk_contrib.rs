//! Marginal and total risk contribution per position.
//!
//! The doctrine this serves (docs/PORTFOLIO.md): a position is judged by its
//! *risk contribution*, not its weight. A 4% weight in a high-idiosyncratic
//! name can be more dangerous than an 8% weight in a stable compounder, and
//! only this decomposition can see that.

use super::covariance::RiskModel;
use super::errors::PortfolioMathError;

/// Euler decomposition of portfolio volatility.
#[derive(Debug, Clone)]
pub struct RiskContributions {
    /// Annualized portfolio volatility (sigma * sqrt(periods_per_year)).
    pub total_vol_annualized: f64,
    /// Marginal contribution per asset: (Sigma w)_i / sigma (per-period).
    pub marginal: Vec<f64>,
    /// Absolute contribution per asset: w_i * marginal_i. Sums to sigma.
    pub contribution: Vec<f64>,
    /// Fractional contribution per asset. Sums to 1.
    pub pct_contribution: Vec<f64>,
}

/// Compute risk contributions of `weights` under `model`.
///
/// Refuses non-finite weights, shape mismatches, and a zero-volatility
/// portfolio (the decomposition is undefined there).
pub fn risk_contributions(
    model: &RiskModel,
    weights: &[f64],
) -> Result<RiskContributions, PortfolioMathError> {
    let n = model.n_assets;
    if weights.len() != n {
        return Err(PortfolioMathError::ShapeMismatch(format!(
            "{} weights for {n} assets",
            weights.len()
        )));
    }
    for (i, w) in weights.iter().enumerate() {
        if !w.is_finite() {
            return Err(PortfolioMathError::NonFinite { row: 0, col: i });
        }
    }

    // sigma^2 = w' Sigma w ; cov_w = Sigma w
    let cov_w: Vec<f64> = model
        .cov
        .chunks_exact(n)
        .map(|row| row.iter().zip(weights).map(|(c, w)| c * w).sum())
        .collect();
    let variance: f64 = weights.iter().zip(cov_w.iter()).map(|(w, c)| w * c).sum();
    if variance <= 0.0 {
        return Err(PortfolioMathError::DegenerateInput(format!(
            "portfolio variance {variance:.3e} is not positive; risk decomposition undefined"
        )));
    }
    let sigma = variance.sqrt();

    let marginal: Vec<f64> = cov_w.iter().map(|c| c / sigma).collect();
    let contribution: Vec<f64> = weights.iter().zip(marginal.iter()).map(|(w, m)| w * m).collect();
    let pct_contribution: Vec<f64> = contribution.iter().map(|c| c / sigma).collect();

    Ok(RiskContributions {
        total_vol_annualized: sigma * model.periods_per_year.sqrt(),
        marginal,
        contribution,
        pct_contribution,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_2(cov: [f64; 4]) -> RiskModel {
        RiskModel {
            cov: cov.to_vec(),
            n_assets: 2,
            asset_ids: vec!["A".into(), "B".into()],
            periods_per_year: 252.0,
            shrinkage_intensity: 0.0,
            n_obs: 100,
        }
    }

    #[test]
    fn contributions_sum_to_sigma() {
        let m = model_2([0.04, 0.01, 0.01, 0.09]);
        let rc = risk_contributions(&m, &[0.6, 0.4]).unwrap();
        let sigma = rc.total_vol_annualized / 252.0_f64.sqrt();
        let sum: f64 = rc.contribution.iter().sum();
        assert!((sum - sigma).abs() < 1e-12);
        let pct_sum: f64 = rc.pct_contribution.iter().sum();
        assert!((pct_sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn symmetric_assets_split_evenly() {
        let m = model_2([0.04, 0.01, 0.01, 0.04]);
        let rc = risk_contributions(&m, &[0.5, 0.5]).unwrap();
        assert!((rc.pct_contribution[0] - 0.5).abs() < 1e-12);
        assert!((rc.pct_contribution[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn zero_weight_asset_has_zero_contribution_but_nonzero_marginal() {
        let m = model_2([0.04, 0.01, 0.01, 0.09]);
        let rc = risk_contributions(&m, &[1.0, 0.0]).unwrap();
        assert_eq!(rc.contribution[1], 0.0);
        assert!(rc.marginal[1] > 0.0); // correlated asset still has marginal risk
    }

    #[test]
    fn refuses_zero_portfolio() {
        let m = model_2([0.04, 0.01, 0.01, 0.09]);
        assert!(matches!(
            risk_contributions(&m, &[0.0, 0.0]),
            Err(PortfolioMathError::DegenerateInput(_))
        ));
    }

    #[test]
    fn refuses_shape_and_nan() {
        let m = model_2([0.04, 0.01, 0.01, 0.09]);
        assert!(risk_contributions(&m, &[0.5]).is_err());
        assert!(risk_contributions(&m, &[0.5, f64::NAN]).is_err());
    }
}
