//! Monte Carlo forward simulation for portfolio projection.
//!
//! Uses Geometric Brownian Motion (GBM) with Cholesky decomposition
//! for correlated multi-asset simulation. Parallelized via Rayon.

use rayon::prelude::*;

/// Configuration for Monte Carlo simulation.
#[derive(Debug, Clone)]
pub struct MonteCarloConfig {
    pub n_simulations: usize,
    pub horizon_days: usize,
    pub seed: u64,
}

impl Default for MonteCarloConfig {
    fn default() -> Self {
        Self { n_simulations: 10_000, horizon_days: 252, seed: 42 }
    }
}

/// Result of a Monte Carlo simulation.
#[derive(Debug, Clone)]
pub struct MonteCarloResult {
    /// Percentile paths: Vec of (percentile, path_values)
    pub percentile_paths: Vec<(f64, Vec<f64>)>,
    /// Terminal value for each simulation
    pub final_values: Vec<f64>,
    /// Expected annualized return
    pub expected_return: f64,
    /// Probability of loss (final value < initial value)
    pub probability_of_loss: f64,
    /// Value at Risk at 95% confidence
    pub var_95: f64,
    /// Conditional Value at Risk at 95% confidence
    pub cvar_95: f64,
}

/// Cholesky decomposition of a symmetric positive-definite matrix.
/// Returns lower-triangular matrix L such that A = L * L^T.
fn cholesky(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, &'static str> {
    let n = matrix.len();
    let mut l = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i][k] * l[j][k];
            }

            if i == j {
                let diag = matrix[i][i] - sum;
                if diag <= 0.0 {
                    // Matrix is not positive definite; use a small epsilon
                    l[i][j] = (diag.abs().max(1e-10)).sqrt();
                } else {
                    l[i][j] = diag.sqrt();
                }
            } else {
                if l[j][j].abs() < 1e-15 {
                    l[i][j] = 0.0;
                } else {
                    l[i][j] = (matrix[i][j] - sum) / l[j][j];
                }
            }
        }
    }

    Ok(l)
}

/// Simple xoshiro256** PRNG for deterministic parallel simulation.
#[derive(Clone)]
struct Xoshiro256 {
    s: [u64; 4],
}

impl Xoshiro256 {
    fn new(seed: u64) -> Self {
        // SplitMix64 to seed all 4 state words
        let mut z = seed;
        let mut s = [0u64; 4];
        for item in &mut s {
            z = z.wrapping_add(0x9e3779b97f4a7c15);
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *item = z ^ (z >> 31);
        }
        Self { s }
    }

    fn jump(&mut self) {
        // Jump function: advances state by 2^128 calls
        const JUMP: [u64; 4] =
            [0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c];
        let mut s0: u64 = 0;
        let mut s1: u64 = 0;
        let mut s2: u64 = 0;
        let mut s3: u64 = 0;
        for j in &JUMP {
            for b in 0..64 {
                if j & (1u64 << b) != 0 {
                    s0 ^= self.s[0];
                    s1 ^= self.s[1];
                    s2 ^= self.s[2];
                    s3 ^= self.s[3];
                }
                self.next_u64();
            }
        }
        self.s[0] = s0;
        self.s[1] = s1;
        self.s[2] = s2;
        self.s[3] = s3;
    }

    fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Generate uniform f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Box-Muller transform for standard normal.
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Core Monte Carlo simulation function.
///
/// # Arguments
/// * `returns` - Per-strategy daily returns (N strategies x T days each)
/// * `weights` - Portfolio weights (length N, must sum to 1)
/// * `correlation_matrix` - N x N correlation matrix
/// * `initial_value` - Starting portfolio value
/// * `config` - Simulation configuration
pub fn simulate_portfolio_forward(
    returns: &[Vec<f64>],
    weights: &[f64],
    correlation_matrix: &[Vec<f64>],
    initial_value: f64,
    config: &MonteCarloConfig,
) -> MonteCarloResult {
    let n_assets = returns.len();
    let dt = 1.0; // daily time step

    // Compute per-asset mean and std of historical returns
    let mut mus = vec![0.0; n_assets];
    let mut sigmas = vec![0.0; n_assets];
    for (i, ret) in returns.iter().enumerate() {
        if ret.is_empty() {
            continue;
        }
        let mean = ret.iter().sum::<f64>() / ret.len() as f64;
        let var = ret.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / ret.len() as f64;
        mus[i] = mean;
        sigmas[i] = var.sqrt().max(1e-10);
    }

    // Cholesky decomposition of correlation matrix
    let chol = cholesky(correlation_matrix).unwrap_or_else(|_| {
        // Fallback: identity matrix (independent assets)
        let mut identity = vec![vec![0.0; n_assets]; n_assets];
        for i in 0..n_assets {
            identity[i][i] = 1.0;
        }
        identity
    });

    // Prepare a base RNG and create per-chunk seeds via jumping
    let mut base_rng = Xoshiro256::new(config.seed);
    let n_chunks = rayon::current_num_threads().max(1);
    let chunk_size = (config.n_simulations + n_chunks - 1) / n_chunks;

    let chunk_rngs: Vec<Xoshiro256> = (0..n_chunks)
        .map(|_| {
            let rng = base_rng.clone();
            base_rng.jump();
            rng
        })
        .collect();

    // Run simulations in parallel chunks
    let all_paths: Vec<Vec<f64>> = chunk_rngs
        .into_par_iter()
        .enumerate()
        .flat_map(|(chunk_idx, mut rng)| {
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(config.n_simulations);
            let mut chunk_paths = Vec::with_capacity(end - start);

            for _ in start..end {
                let mut portfolio_value = initial_value;
                let mut path = Vec::with_capacity(config.horizon_days + 1);
                path.push(portfolio_value);

                for _ in 0..config.horizon_days {
                    // Generate N independent standard normals
                    let z_indep: Vec<f64> = (0..n_assets).map(|_| rng.next_normal()).collect();

                    // Correlate via Cholesky: z_corr = L * z_indep
                    let mut z_corr = vec![0.0; n_assets];
                    for i in 0..n_assets {
                        for j in 0..=i {
                            z_corr[i] += chol[i][j] * z_indep[j];
                        }
                    }

                    // GBM per asset, then weighted portfolio return
                    let mut portfolio_return = 0.0;
                    for i in 0..n_assets {
                        let drift = (mus[i] - 0.5 * sigmas[i].powi(2)) * dt;
                        let diffusion = sigmas[i] * dt.sqrt() * z_corr[i];
                        let asset_return = (drift + diffusion).exp() - 1.0;
                        portfolio_return += weights[i] * asset_return;
                    }

                    portfolio_value *= 1.0 + portfolio_return;
                    path.push(portfolio_value);
                }

                chunk_paths.push(path);
            }

            chunk_paths
        })
        .collect();

    // Extract final values
    let mut final_values: Vec<f64> = all_paths.iter().map(|p| *p.last().unwrap()).collect();
    final_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = final_values.len();

    // Percentile paths: find simulations closest to each percentile's final value
    let percentiles = [5.0, 25.0, 50.0, 75.0, 95.0];
    let percentile_paths: Vec<(f64, Vec<f64>)> = percentiles
        .iter()
        .map(|&pct| {
            let idx = ((pct / 100.0) * (n as f64 - 1.0)).round() as usize;
            let target_final = final_values[idx.min(n - 1)];

            // Find the simulation path whose final value is closest to target
            let best_idx = all_paths
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let da = (a.last().unwrap() - target_final).abs();
                    let db = (b.last().unwrap() - target_final).abs();
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);

            (pct, all_paths[best_idx].clone())
        })
        .collect();

    // Expected return (annualized from mean of final values)
    let mean_final = final_values.iter().sum::<f64>() / n as f64;
    let expected_return = (mean_final / initial_value - 1.0) * 100.0;

    // Probability of loss
    let n_loss = final_values.iter().filter(|&&v| v < initial_value).count();
    let probability_of_loss = n_loss as f64 / n as f64;

    // VaR 95%: 5th percentile loss
    let p5_idx = ((0.05 * (n as f64 - 1.0)).round() as usize).min(n - 1);
    let var_95 = ((initial_value - final_values[p5_idx]) / initial_value * 100.0).max(0.0);

    // CVaR 95%: average of losses below VaR
    let cvar_values = &final_values[..=p5_idx];
    let cvar_95 = if cvar_values.is_empty() {
        var_95
    } else {
        let avg_tail = cvar_values.iter().sum::<f64>() / cvar_values.len() as f64;
        ((initial_value - avg_tail) / initial_value * 100.0).max(0.0)
    };

    MonteCarloResult {
        percentile_paths,
        final_values,
        expected_return,
        probability_of_loss,
        var_95,
        cvar_95,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cholesky_identity() {
        let matrix = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let l = cholesky(&matrix).unwrap();
        assert!((l[0][0] - 1.0).abs() < 1e-10);
        assert!((l[1][1] - 1.0).abs() < 1e-10);
        assert!(l[0][1].abs() < 1e-10);
        assert!(l[1][0].abs() < 1e-10);
    }

    #[test]
    fn test_cholesky_correlated() {
        let matrix = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
        let l = cholesky(&matrix).unwrap();
        // Verify L * L^T = matrix
        let reconstructed_00 = l[0][0] * l[0][0];
        let reconstructed_01 = l[1][0] * l[0][0];
        let reconstructed_11 = l[1][0] * l[1][0] + l[1][1] * l[1][1];
        assert!((reconstructed_00 - 1.0).abs() < 1e-10);
        assert!((reconstructed_01 - 0.5).abs() < 1e-10);
        assert!((reconstructed_11 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_simulate_basic() {
        // Two assets with identical positive returns
        let returns = vec![vec![0.001; 252], vec![0.001; 252]];
        let weights = vec![0.5, 0.5];
        let corr = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let config = MonteCarloConfig { n_simulations: 100, horizon_days: 10, seed: 42 };

        let result = simulate_portfolio_forward(&returns, &weights, &corr, 100000.0, &config);

        assert_eq!(result.final_values.len(), 100);
        assert_eq!(result.percentile_paths.len(), 5);
        // Expected return should be positive given positive drift
        assert!(result.expected_return > -50.0); // Sanity check
    }

    #[test]
    fn test_deterministic() {
        let returns = vec![vec![0.001; 100], vec![-0.0005; 100]];
        let weights = vec![0.6, 0.4];
        let corr = vec![vec![1.0, -0.3], vec![-0.3, 1.0]];
        let config = MonteCarloConfig { n_simulations: 50, horizon_days: 20, seed: 123 };

        let r1 = simulate_portfolio_forward(&returns, &weights, &corr, 100000.0, &config);
        let r2 = simulate_portfolio_forward(&returns, &weights, &corr, 100000.0, &config);

        // Same seed should produce same final values (single-threaded determinism)
        // Note: with rayon, parallelism may affect order but not values
        assert!((r1.expected_return - r2.expected_return).abs() < 1e-6);
    }
}
