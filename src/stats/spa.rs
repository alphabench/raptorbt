//! Hansen's superior predictive ability (SPA) and White's reality check: whether the best of a
//! set of strategies beats a benchmark after allowing for the search across all of them.
//!
//! This is `arch.bootstrap.SPA` as arch 8.0.0 runs it. The statistic is the largest mean excess
//! return, **not studentised**: arch's `studentize` flag does nothing in that release, and
//! replaying its stored bootstrap indices without studentisation reproduces its p-values
//! exactly. The variance estimate (Politis-Romano stationary-bootstrap weights) is used only to
//! decide which strategies are recentred for the "consistent" p-value.
//!
//! The three p-values differ in how the bootstrap means are centred: `lower` leaves the
//! strategies that are worse than the benchmark where they are, `consistent` recentres only
//! those whose mean is not significantly negative, `upper` (White) recentres all.
//!
//! [`spa_replay`] takes the bootstrap indices (arch's, in the tests). [`spa_test`] draws its own:
//! each repetition has an independent SplitMix64 stream whose state is the repetition's draw
//! from a master stream seeded with `seed`, so the answer is the same on any number of threads.
//! It uses the stationary-bootstrap scheme of `research::stationary_bootstrap` (a block starts
//! at a random period and runs on, wrapping, with probability `1 - 1/block` per step), but shares
//! its indices across the strategy columns, which that function's per-series sums cannot.

use rayon::prelude::*;

use crate::research::SplitMix64;

use super::error::StatsError;
use super::hac::{finite_series, peak_to_peak};

pub const MIN_SPA_OBS: usize = 60;
pub const MAX_STRATEGIES: usize = 200;
pub const MIN_REPS: usize = 200;
pub const MAX_REPS: usize = 5000;

/// The three p-values: arch's `lower`, `consistent` and `upper` (White's reality check).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpaPvalues {
    pub lower: f64,
    pub consistent: f64,
    pub upper: f64,
}

/// What a replay of given bootstrap indices produced.
#[derive(Debug, Clone, PartialEq)]
pub struct SpaReplay {
    pub pvalues: SpaPvalues,
    /// Mean of each strategy's excess return over the benchmark (arch's `loss_diff` mean).
    pub loss_diff_mean: Vec<f64>,
    /// Long-run variance of each excess-return series (stationary-bootstrap weights).
    pub loss_diff_var: Vec<f64>,
    /// Whether each strategy is recentred in the consistent p-value.
    pub valid: Vec<bool>,
    /// Per repetition, the largest recentred bootstrap mean over strategies, for the lower,
    /// consistent and upper centring.
    pub max_simulated: Vec<[f64; 3]>,
}

/// The result of [`spa_test`], full precision. `best_strategy` is the column index.
#[derive(Debug, Clone, PartialEq)]
pub struct Spa {
    pub n_periods: usize,
    pub n_strategies: usize,
    pub best_strategy: usize,
    pub best_mean_excess: f64,
    pub best_per_period_sharpe_excess: Option<f64>,
    pub p_value_spa_consistent: f64,
    pub p_value_spa_lower: f64,
    pub p_value_white_reality_check: f64,
    pub mean_block: usize,
    pub reps: usize,
    pub seed: u64,
}

fn seq_mean(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for v in x {
        s += v;
    }
    s / x.len() as f64
}

/// Long-run variance of a demeaned series with the stationary-bootstrap weights.
fn long_run_variance(d: &[f64], kappa: &[f64]) -> f64 {
    let t = d.len();
    let tf = t as f64;
    let mut var = d.iter().map(|x| x * x).sum::<f64>() / tf;
    for (i, kap) in kappa.iter().enumerate().skip(1) {
        let mut s = 0.0;
        for u in 0..t - i {
            s += d[u] * d[u + i];
        }
        var += 2.0 * kap * s / tf;
    }
    var
}

/// Bootstrap p-values for excess returns `loss_diff` (one column per strategy, the strategy minus
/// the benchmark) over the given resamples: `indices[r]` are the periods of repetition `r`.
pub fn spa_replay<C: AsRef<[f64]> + Sync, I: AsRef<[u32]> + Sync>(
    loss_diff: &[C],
    block: usize,
    indices: &[I],
) -> Result<SpaReplay, StatsError> {
    let t = loss_diff.first().map_or(0, |c| c.as_ref().len());
    if t == 0 || block == 0 || indices.is_empty() {
        return Err(StatsError::Invalid("spa: nothing to resample".into()));
    }
    if loss_diff.iter().any(|c| c.as_ref().len() != t) {
        return Err(StatsError::Invalid("spa: the columns differ in length".into()));
    }
    if indices.iter().any(|r| r.as_ref().iter().any(|&i| i as usize >= t)) {
        return Err(StatsError::Invalid("spa: a bootstrap index is out of range".into()));
    }
    let tf = t as f64;
    let p = 1.0 / block as f64;
    let kappa: Vec<f64> = (0..t)
        .map(|i| {
            let r = i as f64 / tf;
            (1.0 - r) * (1.0 - p).powf(i as f64) + r * (1.0 - p).powf((t - i) as f64)
        })
        .collect();

    let means: Vec<f64> = loss_diff.iter().map(|c| seq_mean(c.as_ref())).collect();
    let var: Vec<f64> = loss_diff
        .par_iter()
        .zip(&means)
        .map(|(c, m)| {
            let d: Vec<f64> = c.as_ref().iter().map(|x| x - m).collect();
            long_run_variance(&d, &kappa)
        })
        .collect();
    let threshold_scale = 2.0 * tf.ln().ln();
    let valid: Vec<bool> =
        means.iter().zip(&var).map(|(m, v)| *m >= -((v / tf) * threshold_scale).sqrt()).collect();

    // Upper always recentres; consistent skips the invalid columns; lower leaves the worse alone.
    let centres: [Vec<f64>; 3] = [
        means.iter().map(|&m| if m < 0.0 { 0.0 } else { m }).collect(),
        means.iter().zip(&valid).map(|(&m, &ok)| if ok { m } else { 0.0 }).collect(),
        means.clone(),
    ];

    let max_simulated: Vec<[f64; 3]> = indices
        .par_iter()
        .map(|rep| {
            let rep = rep.as_ref();
            let mut best = [f64::NEG_INFINITY; 3];
            for (j, c) in loss_diff.iter().enumerate() {
                let c = c.as_ref();
                let mut s = 0.0;
                for &i in rep {
                    s += c[i as usize];
                }
                let star = s / rep.len() as f64;
                for v in 0..3 {
                    best[v] = best[v].max(star - centres[v][j]);
                }
            }
            best
        })
        .collect();

    let observed = means.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let reps = max_simulated.len() as f64;
    let share = |v: usize| max_simulated.iter().filter(|m| m[v] > observed).count() as f64 / reps;
    Ok(SpaReplay {
        pvalues: SpaPvalues { lower: share(0), consistent: share(1), upper: share(2) },
        loss_diff_mean: means,
        loss_diff_var: var,
        valid,
        max_simulated,
    })
}

/// Whether the best strategy beats `benchmark` (zero when `None`) after allowing for having
/// searched across all columns of `cols`. Returns are per period. `block_size` is the mean block
/// of the stationary bootstrap, default `max(2, round(n^(1/3)))`.
pub fn spa_test<C: AsRef<[f64]> + Sync>(
    cols: &[C],
    benchmark: Option<&[f64]>,
    reps: usize,
    block_size: Option<i64>,
    seed: u64,
) -> Result<Spa, StatsError> {
    if cols.is_empty() {
        return Err(StatsError::InsufficientObservations {
            what: "strategies".into(),
            found: 0,
            needed: 1,
        });
    }
    let series = cols
        .iter()
        .enumerate()
        .map(|(i, c)| finite_series(c.as_ref(), &format!("strategies[{i}]"), MIN_SPA_OBS))
        .collect::<Result<Vec<_>, _>>()?;
    let n = series[0].len();
    if series.iter().any(|s| s.len() != n) {
        let mut lens: Vec<usize> = series.iter().map(|s| s.len()).collect();
        lens.sort_unstable();
        lens.dedup();
        return Err(StatsError::Invalid(format!(
            "strategies: the series differ in length ({lens:?})"
        )));
    }
    let k = series.len();
    if k > MAX_STRATEGIES {
        return Err(StatsError::Invalid(format!("{k} strategies, at most {MAX_STRATEGIES}")));
    }
    let bench: Vec<f64> = match benchmark {
        None => vec![0.0; n],
        Some(b) => {
            let b = finite_series(b, "benchmark", n)?;
            if b.len() != n {
                return Err(StatsError::Invalid(
                    "benchmark and strategies differ in length".into(),
                ));
            }
            b.to_vec()
        }
    };
    if !(MIN_REPS..=MAX_REPS).contains(&reps) {
        return Err(StatsError::Invalid(format!("reps must be {MIN_REPS} to {MAX_REPS}")));
    }
    let block =
        block_size.unwrap_or_else(|| ((n as f64).powf(1.0 / 3.0).round_ties_even() as i64).max(2));
    if !(1 <= block && (block as usize) < n / 2) {
        return Err(StatsError::Invalid(format!("block_size must be 1 to {}", n / 2 - 1)));
    }
    let block = block as usize;

    let diff: Vec<Vec<f64>> =
        series.iter().map(|s| s.iter().zip(&bench).map(|(a, b)| a - b).collect()).collect();

    // A constant excess has no sampling variation to bootstrap: the count of resamples strictly
    // above the observed mean would be zero and read as p = 0 ("beats the benchmark at any level").
    if diff.iter().all(|d| peak_to_peak(d) == 0.0) {
        return Err(StatsError::DegenerateSeries(
            "every strategy's excess over the benchmark is constant (identical to it, or all zero): \
             there is no evidence to test"
                .into(),
        ));
    }

    // One independent stream per repetition, started from a draw of the master stream.
    let mut master = SplitMix64::new(seed);
    let starts: Vec<u64> = (0..reps).map(|_| master.next_u64()).collect();
    let p_new = 1.0 / block as f64;
    let len = n as u64;
    let indices: Vec<Vec<u32>> = starts
        .par_iter()
        .map(|&s| {
            let mut rng = SplitMix64::new(s);
            let mut i = rng.below(len) as usize;
            let mut rep = Vec::with_capacity(n);
            for _ in 0..n {
                rep.push(i as u32);
                i = if rng.uniform() < p_new { rng.below(len) as usize } else { (i + 1) % n };
            }
            rep
        })
        .collect();
    let replay = spa_replay(&diff, block, &indices)?;

    let mut best = 0;
    for (j, m) in replay.loss_diff_mean.iter().enumerate() {
        if *m > replay.loss_diff_mean[best] {
            best = j;
        }
    }
    let ex = &diff[best];
    let mean = seq_mean(ex);
    let sd0 = (ex.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64).sqrt();
    let sd1 = (ex.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n as f64 - 1.0)).sqrt();
    let sharpe = if sd0 > 0.0 { Some(mean / sd1) } else { None };
    Ok(Spa {
        n_periods: n,
        n_strategies: k,
        best_strategy: best,
        best_mean_excess: mean,
        best_per_period_sharpe_excess: sharpe,
        p_value_spa_consistent: replay.pvalues.consistent,
        p_value_spa_lower: replay.pvalues.lower,
        p_value_white_reality_check: replay.pvalues.upper,
        mean_block: block,
        reps,
        seed,
    })
}
