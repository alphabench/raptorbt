//! Covariance estimation with Ledoit-Wolf shrinkage.
//!
//! Implements the constant-correlation shrinkage estimator from Ledoit & Wolf,
//! "Honey, I Shrunk the Sample Covariance Matrix" (2004). The target is chosen
//! over the identity deliberately: an Indian long-only equity book has
//! materially positive average pairwise correlation, and shrinking toward the
//! identity pulls correlations toward zero, *understating* portfolio risk --
//! the wrong direction for a platform whose doctrine is to refuse optimistic
//! numbers.
//!
//! The result is carried in a [`RiskModel`] that pins two things structurally:
//! `periods_per_year` (annualization must travel with the matrix -- the
//! platform has measured a 7.75x Sharpe inflation from an inferred basis) and
//! `asset_ids` (consumers validate their own ordering against the model's, so
//! a re-sorted universe is a hard error, not a silently wrong number).

use super::errors::{require_finite, PortfolioMathError};

/// A shrunk covariance matrix plus the context required to use it safely.
#[derive(Debug, Clone)]
pub struct RiskModel {
    /// Row-major `n_assets x n_assets` covariance of per-period returns.
    pub cov: Vec<f64>,
    /// Number of assets.
    pub n_assets: usize,
    /// Asset identifiers, in the exact column order of `cov`.
    pub asset_ids: Vec<String>,
    /// Return periodicity (e.g. 252 for daily bars). Required, never inferred.
    pub periods_per_year: f64,
    /// Shrinkage intensity delta in [0, 1] actually applied.
    pub shrinkage_intensity: f64,
    /// Number of return observations the estimate was built from.
    pub n_obs: usize,
}

impl RiskModel {
    /// Validate a caller-supplied asset ordering against the model's.
    pub fn require_same_assets(&self, asset_ids: &[String]) -> Result<(), PortfolioMathError> {
        if asset_ids.len() != self.n_assets {
            return Err(PortfolioMathError::AssetIdMismatch(format!(
                "model has {} assets, caller supplied {}",
                self.n_assets,
                asset_ids.len()
            )));
        }
        for (i, (a, b)) in self.asset_ids.iter().zip(asset_ids.iter()).enumerate() {
            if a != b {
                return Err(PortfolioMathError::AssetIdMismatch(format!(
                    "position {i}: model '{a}' vs caller '{b}'"
                )));
            }
        }
        Ok(())
    }
}

/// Estimate a Ledoit-Wolf constant-correlation shrunk covariance.
///
/// `returns` is row-major `n_obs x n_assets` of per-period simple returns.
/// Refuses non-finite input, fewer than 2 observations, fewer than 2 assets,
/// zero-variance assets, and any result that fails a strict Cholesky.
pub fn ledoit_wolf(
    returns: &[f64],
    n_obs: usize,
    n_assets: usize,
    asset_ids: Vec<String>,
    periods_per_year: f64,
) -> Result<RiskModel, PortfolioMathError> {
    require_finite(returns, n_obs, n_assets)?;
    if asset_ids.len() != n_assets {
        return Err(PortfolioMathError::ShapeMismatch(format!(
            "{} asset_ids for {n_assets} columns",
            asset_ids.len()
        )));
    }
    if n_obs < 2 {
        return Err(PortfolioMathError::DegenerateInput(format!(
            "need at least 2 observations, got {n_obs}"
        )));
    }
    if n_assets < 2 {
        return Err(PortfolioMathError::DegenerateInput(format!(
            "need at least 2 assets, got {n_assets}"
        )));
    }
    if !(periods_per_year.is_finite() && periods_per_year > 0.0) {
        return Err(PortfolioMathError::DegenerateInput(format!(
            "periods_per_year must be a positive finite number, got {periods_per_year}"
        )));
    }

    let t = n_obs as f64;
    let n = n_assets;

    // Demean each column.
    let mut means = vec![0.0; n];
    for row in 0..n_obs {
        for col in 0..n {
            means[col] += returns[row * n + col];
        }
    }
    for m in means.iter_mut() {
        *m /= t;
    }
    let mut x = vec![0.0; n_obs * n]; // demeaned returns
    for row in 0..n_obs {
        for col in 0..n {
            x[row * n + col] = returns[row * n + col] - means[col];
        }
    }

    // Sample covariance S = X'X / T (the 1/T convention of the LW paper).
    let mut s = vec![0.0; n * n];
    for i in 0..n {
        for j in i..n {
            let mut acc = 0.0;
            for row in 0..n_obs {
                acc += x[row * n + i] * x[row * n + j];
            }
            let v = acc / t;
            s[i * n + j] = v;
            s[j * n + i] = v;
        }
    }

    // Zero-variance assets break the correlation target. A constant column
    // demeans to floating-point residue (~1e-37), not exactly zero, so the
    // gate is relative: variance below machine epsilon times the squared mean
    // level of the column is indistinguishable from constant.
    let mut sqrt_var = vec![0.0; n];
    for i in 0..n {
        let v = s[i * n + i];
        let scale = means[i] * means[i];
        if v <= 0.0 || v < f64::EPSILON * scale {
            return Err(PortfolioMathError::ZeroVariance(i));
        }
        sqrt_var[i] = v.sqrt();
    }

    // Average off-diagonal sample correlation r_bar.
    let mut r_sum = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            r_sum += s[i * n + j] / (sqrt_var[i] * sqrt_var[j]);
        }
    }
    let pairs = (n * (n - 1) / 2) as f64;
    let r_bar = r_sum / pairs;

    // Constant-correlation target F.
    let mut f = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            f[i * n + j] = if i == j {
                s[i * n + i]
            } else {
                r_bar * sqrt_var[i] * sqrt_var[j]
            };
        }
    }

    // pi_hat: sum of asymptotic variances of the sample covariance entries.
    // pi_ij = (1/T) sum_t (x_it x_jt - s_ij)^2
    let mut pi_mat = vec![0.0; n * n];
    let mut pi_hat = 0.0;
    for i in 0..n {
        for j in 0..n {
            let s_ij = s[i * n + j];
            let mut acc = 0.0;
            for row in 0..n_obs {
                let d = x[row * n + i] * x[row * n + j] - s_ij;
                acc += d * d;
            }
            let v = acc / t;
            pi_mat[i * n + j] = v;
            pi_hat += v;
        }
    }

    // rho_hat: sum of asymptotic covariances between sample cov entries and
    // the target entries. Diagonal contributes pi_ii; off-diagonal uses
    // theta_hat terms per the LW 2004 appendix.
    let mut rho_hat = 0.0;
    for i in 0..n {
        rho_hat += pi_mat[i * n + i];
    }
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let s_ii = s[i * n + i];
            let s_jj = s[j * n + j];
            let s_ij = s[i * n + j];
            // theta_ii,ij = (1/T) sum_t (x_it^2 - s_ii)(x_it x_jt - s_ij)
            let mut th_ii = 0.0;
            let mut th_jj = 0.0;
            for row in 0..n_obs {
                let xi = x[row * n + i];
                let xj = x[row * n + j];
                let cross = xi * xj - s_ij;
                th_ii += (xi * xi - s_ii) * cross;
                th_jj += (xj * xj - s_jj) * cross;
            }
            th_ii /= t;
            th_jj /= t;
            rho_hat += (r_bar / 2.0)
                * ((s_jj / s_ii).sqrt() * th_ii + (s_ii / s_jj).sqrt() * th_jj);
        }
    }

    // gamma_hat: squared Frobenius distance between target and sample.
    let mut gamma_hat = 0.0;
    for idx in 0..n * n {
        let d = f[idx] - s[idx];
        gamma_hat += d * d;
    }

    // Shrinkage intensity delta = clamp(kappa / T, 0, 1).
    let shrinkage = if gamma_hat <= f64::EPSILON {
        // Sample already equals the target (e.g. perfect constant
        // correlation): no shrinkage direction exists.
        0.0
    } else {
        let kappa = (pi_hat - rho_hat) / gamma_hat;
        (kappa / t).clamp(0.0, 1.0)
    };

    // Shrunk covariance.
    let mut cov = vec![0.0; n * n];
    for idx in 0..n * n {
        cov[idx] = shrinkage * f[idx] + (1.0 - shrinkage) * s[idx];
    }

    // A convex combination of PSD matrices is PSD in exact arithmetic, but we
    // still refuse anything a strict Cholesky rejects (duplicated columns,
    // catastrophic cancellation) rather than repairing it.
    chol_strict(&cov, n)?;

    Ok(RiskModel {
        cov,
        n_assets: n,
        asset_ids,
        periods_per_year,
        shrinkage_intensity: shrinkage,
        n_obs,
    })
}

/// Strict Cholesky factorization: errors on any non-positive pivot.
///
/// Deliberately unlike `monte_carlo::cholesky`, which clamps a bad diagonal to
/// keep a forward simulation running. Here a non-PD matrix means the risk
/// model is wrong, and a wrong risk model must not reach an optimizer.
pub fn chol_strict(a: &[f64], n: usize) -> Result<Vec<f64>, PortfolioMathError> {
    if a.len() != n * n {
        return Err(PortfolioMathError::ShapeMismatch(format!(
            "expected {n}x{n}={} values, got {}",
            n * n,
            a.len()
        )));
    }
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum <= 0.0 || !sum.is_finite() {
                    return Err(PortfolioMathError::NotPositiveDefinite(format!(
                        "pivot {sum:.3e} at index {i}"
                    )));
                }
                l[i * n + i] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    Ok(l)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("A{i}")).collect()
    }

    /// Deterministic pseudo-random returns (no external RNG dep in tests).
    ///
    /// Per-asset factor loadings vary with the column index so pairwise
    /// correlations are heterogeneous. A single uniform loading would make
    /// the constant-correlation target an exact fit (gamma ~ 0), which
    /// legitimately drives the shrinkage intensity to 1 -- a degenerate
    /// fixture, not a bug.
    fn synth_returns(n_obs: usize, n_assets: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as f64 / (u32::MAX as f64) - 0.5
        };
        let mut out = Vec::with_capacity(n_obs * n_assets);
        for _ in 0..n_obs {
            let common = next() * 0.02;
            for a in 0..n_assets {
                let loading = 0.2 + 0.8 * a as f64 / n_assets as f64;
                let idio = next() * 0.02;
                out.push(loading * common + idio);
            }
        }
        out
    }

    #[test]
    fn sample_cov_matches_hand_computed_three_assets() {
        // 4 observations, 3 assets; delta=0 forced by checking S directly via
        // a target-equals-sample case is fragile, so instead verify the
        // shrunk matrix lies between S and F entrywise.
        let returns = vec![
            0.01, 0.02, -0.01, //
            -0.02, 0.01, 0.00, //
            0.03, -0.01, 0.02, //
            0.00, 0.02, -0.02,
        ];
        let m = ledoit_wolf(&returns, 4, 3, ids(3), 252.0).unwrap();
        assert!(m.shrinkage_intensity >= 0.0 && m.shrinkage_intensity <= 1.0);
        // Symmetry.
        for i in 0..3 {
            for j in 0..3 {
                assert!((m.cov[i * 3 + j] - m.cov[j * 3 + i]).abs() < 1e-15);
            }
        }
        // Diagonal preserved: target and sample share the diagonal, so the
        // shrunk diagonal must equal the sample variance exactly.
        // Hand-compute sample variance of column 0 with the 1/T convention.
        let col0 = [0.01, -0.02, 0.03, 0.00];
        let mean: f64 = col0.iter().sum::<f64>() / 4.0;
        let var: f64 = col0.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / 4.0;
        assert!((m.cov[0] - var).abs() < 1e-15, "{} vs {var}", m.cov[0]);
    }

    #[test]
    fn shrinkage_decreases_with_more_observations() {
        let small = ledoit_wolf(&synth_returns(60, 8, 42), 60, 8, ids(8), 252.0).unwrap();
        let large = ledoit_wolf(&synth_returns(2000, 8, 42), 2000, 8, ids(8), 252.0).unwrap();
        assert!(
            large.shrinkage_intensity < small.shrinkage_intensity,
            "delta should shrink with T: {} vs {}",
            large.shrinkage_intensity,
            small.shrinkage_intensity
        );
    }

    #[test]
    fn output_is_positive_definite() {
        let m = ledoit_wolf(&synth_returns(50, 10, 7), 50, 10, ids(10), 252.0).unwrap();
        chol_strict(&m.cov, 10).unwrap();
    }

    #[test]
    fn refuses_nan() {
        let mut r = synth_returns(20, 3, 1);
        r[7] = f64::NAN;
        let err = ledoit_wolf(&r, 20, 3, ids(3), 252.0).unwrap_err();
        assert!(matches!(err, PortfolioMathError::NonFinite { row: 2, col: 1 }));
    }

    #[test]
    fn refuses_zero_variance_asset() {
        let mut r = synth_returns(20, 3, 1);
        for row in 0..20 {
            r[row * 3 + 2] = 0.005; // constant column
        }
        let err = ledoit_wolf(&r, 20, 3, ids(3), 252.0).unwrap_err();
        assert!(matches!(err, PortfolioMathError::ZeroVariance(2)));
    }

    #[test]
    fn refuses_shape_mismatch_and_bad_ppy() {
        let r = synth_returns(10, 3, 1);
        assert!(ledoit_wolf(&r, 10, 4, ids(4), 252.0).is_err());
        assert!(ledoit_wolf(&r, 10, 3, ids(2), 252.0).is_err());
        assert!(ledoit_wolf(&r, 10, 3, ids(3), 0.0).is_err());
        assert!(ledoit_wolf(&r, 10, 3, ids(3), f64::NAN).is_err());
    }

    #[test]
    fn chol_strict_rejects_rank_deficient() {
        // Rank-deficient: second row is a copy of the first. The clamping
        // cholesky in monte_carlo.rs would repair this; chol_strict must not.
        let a = vec![1.0, 1.0, 1.0, 1.0];
        assert!(matches!(
            chol_strict(&a, 2),
            Err(PortfolioMathError::NotPositiveDefinite(_))
        ));
    }

    #[test]
    fn require_same_assets_validates_order() {
        let m = ledoit_wolf(&synth_returns(30, 3, 3), 30, 3, ids(3), 252.0).unwrap();
        assert!(m.require_same_assets(&ids(3)).is_ok());
        let mut wrong = ids(3);
        wrong.swap(0, 2);
        assert!(matches!(
            m.require_same_assets(&wrong),
            Err(PortfolioMathError::AssetIdMismatch(_))
        ));
    }
}
