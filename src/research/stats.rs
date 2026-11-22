//! The deflated Sharpe ratio, the stationary bootstrap, `without_best` and the Monte Carlo of
//! a finished result's paths (`resample_paths`).

use crate::core::error::{RaptorError, Result};
use crate::math::fsum::fsum;
use crate::pricing::norm::{norm_cdf, norm_ppf};
use crate::stats::StatsError;
use std::f64::consts::EULER_GAMMA as EULER;

use super::SplitMix64;

/// Where the spread of luck's Sharpes comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Luck {
    /// The trials' own spread: right for a search over one family (a parameter
    /// sweep), whose trials scatter around no skill.
    Trials,
    /// The Sharpe's own sampling variance, `1 / (samples - 1)`: right when the trials
    /// are unlike each other and only their number counts.
    Samples,
}

impl Luck {
    pub fn as_str(self) -> &'static str {
        match self {
            Luck::Trials => "trials",
            Luck::Samples => "samples",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "trials" => Ok(Luck::Trials),
            "samples" => Ok(Luck::Samples),
            _ => Err(RaptorError::invalid_parameter("luck is trials or samples")),
        }
    }
}

/// The outcome of [`deflated_sharpe`]. Each variant is one of the three dict shapes
/// the Python binding returns.
#[derive(Debug, Clone, PartialEq)]
pub enum DeflatedSharpe {
    /// Fewer than two trials or three samples: no answer.
    Insufficient { trials: usize },
    /// The skew and kurtosis leave the variance term non-positive.
    MomentsOutOfRange { trials: usize, sr0_annualised: f64 },
    Computed {
        trials: usize,
        luck: Luck,
        sr0_annualised: f64,
        best_annualised: f64,
        dsr: f64,
        z: f64,
        skew: f64,
        kurtosis: f64,
        samples: i64,
    },
}

/// Bailey and López de Prado's deflated Sharpe ratio: the probability that the best
/// annualised Sharpe of `sharpes.len()` trials is above the Sharpe luck alone hands the
/// best of that many trials (`sr0`), given the returns' skew, kurtosis and sample count.
/// Sharpes are annualised with `periods_per_year` and brought back to per-period for the
/// test.
pub fn deflated_sharpe(
    sharpes: &[f64],
    best: f64,
    samples: i64,
    skew: f64,
    kurtosis: f64,
    periods_per_year: f64,
    luck: Luck,
) -> DeflatedSharpe {
    let n = sharpes.len();
    if n < 2 || samples < 3 {
        return DeflatedSharpe::Insufficient { trials: n };
    }
    let per = periods_per_year.sqrt();
    let var = match luck {
        Luck::Samples => 1.0 / (samples - 1) as f64,
        Luck::Trials => {
            let xs: Vec<f64> = sharpes.iter().map(|s| s / per).collect();
            let mean = fsum(xs.iter().copied()) / n as f64;
            fsum(xs.iter().map(|x| (x - mean) * (x - mean))) / (n - 1) as f64
        }
    };
    let nf = n as f64;
    // n >= 2, so both probabilities lie in (0, 1) and the quantiles exist.
    let (Some(q1), Some(q2)) =
        (norm_ppf(1.0 - 1.0 / nf), norm_ppf(1.0 - 1.0 / (nf * std::f64::consts::E)))
    else {
        return DeflatedSharpe::Insufficient { trials: n };
    };
    let sr0 = var.sqrt() * ((1.0 - EULER) * q1 + EULER * q2);
    let sr = best / per;
    let denom = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr * sr;
    if denom <= 0.0 {
        return DeflatedSharpe::MomentsOutOfRange { trials: n, sr0_annualised: sr0 * per };
    }
    let z = (sr - sr0) * ((samples - 1) as f64).sqrt() / denom.sqrt();
    DeflatedSharpe::Computed {
        trials: n,
        luck,
        sr0_annualised: sr0 * per,
        best_annualised: best,
        dsr: norm_cdf(z),
        z,
        skew,
        kurtosis,
        samples,
    }
}

/// The outcome of [`stationary_bootstrap`].
#[derive(Debug, Clone, PartialEq)]
pub enum Bootstrap {
    /// Fewer than two samples, or no draws asked for.
    Insufficient { samples: usize },
    Drawn {
        samples: usize,
        draws: usize,
        block: f64,
        /// 5th, 50th and 95th percentile of the resamples' annualised mean return.
        p05: f64,
        p50: f64,
        p95: f64,
        /// Share of resamples whose annualised mean is above zero.
        above_zero: f64,
    },
}

/// The default mean block of a stationary bootstrap of `n` periods: the cube root of the
/// length, rounded half to even, at least 1.
fn default_block(n: usize) -> f64 {
    (n as f64).powf(1.0 / 3.0).round_ties_even().max(1.0)
}

/// A mean block length is at least one period. NaN is refused too (it would otherwise run as
/// "never start a new block").
fn block_ok(mean_block: f64) -> bool {
    !mean_block.is_nan() && mean_block >= 1.0
}

/// One stationary-bootstrap resample of `n` periods: `visit` is called with each period's
/// index in resample order. The resample starts at a random period and runs on (wrapping),
/// starting a new block at a random period with probability `p_new` after each period.
fn stationary_resample(rng: &mut SplitMix64, n: usize, p_new: f64, mut visit: impl FnMut(usize)) {
    let len = n as u64;
    let mut i = rng.below(len) as usize;
    for _ in 0..n {
        visit(i);
        i = if rng.uniform() < p_new { rng.below(len) as usize } else { (i + 1) % n };
    }
}

/// The nearest-rank `q` quantile of a sorted, non-empty slice: the value at rank
/// `ceil(q * len)`, clamped to the slice.
fn nearest_rank(sorted: &[f64], q: f64) -> f64 {
    let k = (q * sorted.len() as f64).ceil() as i64 - 1;
    sorted[k.clamp(0, sorted.len() as i64 - 1) as usize]
}

/// Politis and Romano's stationary bootstrap of a return series. Each resample is as
/// long as the series, built from blocks that start at a random period and run on
/// (wrapping) for a geometric number of periods of mean `block` (default: the cube root
/// of the length, rounded half to even, at least 1). Deterministic for a seed.
pub fn stationary_bootstrap(
    returns: &[f64],
    draws: usize,
    block: Option<f64>,
    seed: u64,
    periods_per_year: f64,
) -> Result<Bootstrap> {
    let n = returns.len();
    if n < 2 || draws < 1 {
        return Ok(Bootstrap::Insufficient { samples: n });
    }
    let mean_block = block.unwrap_or_else(|| default_block(n));
    if !block_ok(mean_block) {
        return Err(RaptorError::invalid_parameter("block is at least one period"));
    }
    let p_new = 1.0 / mean_block;
    let mut rng = SplitMix64::new(seed);
    let mut means = Vec::with_capacity(draws);
    for _ in 0..draws {
        let mut total = 0.0;
        stationary_resample(&mut rng, n, p_new, |i| total += returns[i]);
        means.push(total / n as f64 * periods_per_year);
    }
    means.sort_by(f64::total_cmp);
    let above = means.iter().filter(|m| **m > 0.0).count();
    Ok(Bootstrap::Drawn {
        samples: n,
        draws,
        block: mean_block,
        p05: nearest_rank(&means, 0.05),
        p50: nearest_rank(&means, 0.50),
        p95: nearest_rank(&means, 0.95),
        above_zero: above as f64 / draws as f64,
    })
}

/// The fewest observations [`resample_paths`] takes. With ten, a stationary bootstrap of the
/// default block (2 periods) has about five blocks to recombine, and an order shuffle has
/// 10! = 3,628,800 orders, far more than the draws (at seven there are 5,040; at six, 720,
/// fewer than the default 2,000 draws, so the same few paths would repeat).
pub const RESAMPLE_MIN_OBSERVATIONS: usize = 10;
/// The most draws [`resample_paths`] takes.
pub const RESAMPLE_MAX_DRAWS: usize = 100_000;
/// The most drawdown levels [`resample_paths`] takes.
pub const RESAMPLE_MAX_LEVELS: usize = 16;
/// The bins of the drawdown histogram [`resample_paths`] returns.
pub const DRAWDOWN_BINS: usize = 20;

/// How [`resample_paths`] builds each path from the observations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Resample {
    /// A return series, resampled by the stationary bootstrap of mean block `block` (`None`:
    /// the default of [`stationary_bootstrap`]).
    Block { block: Option<f64> },
    /// Per-trade returns, drawn with replacement (a bootstrap of trades) or, without, put in
    /// a random order (a shuffle).
    Trades { replace: bool },
}

/// The 5th, 50th, 95th and 99th percentile of a resampled quantity (nearest rank).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Percentiles {
    pub p05: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
}

impl Percentiles {
    fn of(sorted: &[f64], f: impl Fn(f64) -> f64) -> Self {
        Percentiles {
            p05: f(nearest_rank(sorted, 0.05)),
            p50: f(nearest_rank(sorted, 0.50)),
            p95: f(nearest_rank(sorted, 0.95)),
            p99: f(nearest_rank(sorted, 0.99)),
        }
    }

    fn all(v: f64) -> Self {
        Percentiles { p05: v, p50: v, p95: v, p99: v }
    }
}

/// The outcome of [`resample_paths`]. Returns and drawdowns are fractions; a drawdown is
/// positive (0.2 is a fall of 20% from the running peak).
#[derive(Debug, Clone, PartialEq)]
pub struct Paths {
    pub n: usize,
    pub draws: usize,
    /// The mean block used (the block method only).
    pub block: Option<f64>,
    /// Whether the observations were drawn with replacement (the block bootstrap always is).
    pub replace: bool,
    /// The ending return and maximum drawdown of the observations in their own order.
    pub observed_return: f64,
    pub observed_drawdown: f64,
    pub ending_return: Percentiles,
    /// The ending return annualised over `n / periods_per_year` years, when that was given.
    pub annual_return: Option<Percentiles>,
    pub max_drawdown: Percentiles,
    /// Share of paths whose ending return is below zero.
    pub prob_loss: f64,
    /// Each level with the share of paths whose maximum drawdown is at least that level.
    pub prob_drawdown: Vec<(f64, f64)>,
    /// `DRAWDOWN_BINS + 1` edges from 0 to the largest drawn drawdown (to 1 when no path has
    /// one) and the paths in each bin, the last bin closed.
    pub edges: Vec<f64>,
    pub counts: Vec<u64>,
    /// True for a shuffle: every path ends at the observed return.
    pub ending_fixed: bool,
}

/// The log growth of one path and its largest log fall from its running peak, the starting
/// value (log 0) included. `log_growth` holds `ln(1 + r)` per observation.
fn walk(log_growth: &[f64], order: impl Iterator<Item = usize>) -> (f64, f64) {
    let (mut level, mut peak, mut fall) = (0.0f64, 0.0f64, 0.0f64);
    for i in order {
        level += log_growth[i];
        peak = peak.max(level);
        fall = fall.max(peak - level);
    }
    (level, fall)
}

/// A drawdown fraction from a log fall: `1 - e^-fall`.
fn drawdown_of(fall: f64) -> f64 {
    -(-fall).exp_m1()
}

/// A Monte Carlo of a finished result: `draws` paths resampled from `values`, each
/// compounded from a starting value of 1, and the spread of their ending return and maximum
/// drawdown (measured on the compounded path from its running peak, the start included).
///
/// `values` are per-period simple returns (`Resample::Block`, the stationary bootstrap of
/// [`stationary_bootstrap`]) or per-trade returns as fractions of the capital a trade risked
/// (`Resample::Trades`). A shuffle (`replace: false`) keeps every observation once, so every
/// path ends at the observed return and only its drawdown varies. `drawdown_levels` are
/// fractions in (0, 1]. Deterministic for a seed.
pub fn resample_paths(
    values: &[f64],
    method: Resample,
    draws: usize,
    seed: u64,
    periods_per_year: Option<f64>,
    drawdown_levels: &[f64],
) -> std::result::Result<Paths, StatsError> {
    let what = match method {
        Resample::Block { .. } => "returns",
        Resample::Trades { .. } => "trades",
    };
    let n = values.len();
    if n < RESAMPLE_MIN_OBSERVATIONS {
        return Err(StatsError::InsufficientObservations {
            what: what.into(),
            found: n,
            needed: RESAMPLE_MIN_OBSERVATIONS,
        });
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(StatsError::NonFinite { what: what.into() });
    }
    if values.iter().any(|v| *v < -1.0) {
        return Err(StatsError::Invalid(format!(
            "{what} holds a return below -1: a path compounds, and no position loses more than all of it"
        )));
    }
    if !(1..=RESAMPLE_MAX_DRAWS).contains(&draws) {
        return Err(StatsError::Invalid(format!("draws is 1 to {RESAMPLE_MAX_DRAWS}")));
    }
    if drawdown_levels.len() > RESAMPLE_MAX_LEVELS {
        return Err(StatsError::Invalid(format!(
            "drawdown_levels holds at most {RESAMPLE_MAX_LEVELS} levels"
        )));
    }
    if drawdown_levels.iter().any(|l| !(*l > 0.0 && *l <= 1.0)) {
        return Err(StatsError::Invalid("each drawdown level is a fraction in (0, 1]".into()));
    }
    if let Some(ppy) = periods_per_year {
        if !(ppy.is_finite() && ppy > 0.0) {
            return Err(StatsError::Invalid("periods_per_year is a positive number".into()));
        }
    }
    let block = match method {
        Resample::Block { block } => {
            let b = block.unwrap_or_else(|| default_block(n));
            if !block_ok(b) {
                return Err(StatsError::Invalid("block is at least one period".into()));
            }
            Some(b)
        }
        Resample::Trades { .. } => None,
    };

    // ln(1 + r): a path's growth is a sum, so a long path neither overflows nor rounds
    // through thousands of products. A return of -1 is -inf: the path is ruined (a
    // drawdown of 1, an ending return of -1).
    let log_growth: Vec<f64> = values.iter().map(|r| r.ln_1p()).collect();
    let (observed_level, observed_fall) = walk(&log_growth, 0..n);
    let observed_level = if observed_level == f64::NEG_INFINITY {
        observed_level
    } else {
        fsum(log_growth.iter().copied())
    };

    let mut rng = SplitMix64::new(seed);
    let mut levels = Vec::with_capacity(draws);
    let mut falls = Vec::with_capacity(draws);
    match method {
        Resample::Block { .. } => {
            let p_new = 1.0 / block.unwrap_or(1.0);
            let mut order = Vec::with_capacity(n);
            for _ in 0..draws {
                order.clear();
                stationary_resample(&mut rng, n, p_new, |i| order.push(i));
                let (level, fall) = walk(&log_growth, order.iter().copied());
                levels.push(level);
                falls.push(fall);
            }
        }
        Resample::Trades { replace: true } => {
            let len = n as u64;
            let mut order = Vec::with_capacity(n);
            for _ in 0..draws {
                order.clear();
                order.extend((0..n).map(|_| rng.below(len) as usize));
                let (level, fall) = walk(&log_growth, order.iter().copied());
                levels.push(level);
                falls.push(fall);
            }
        }
        Resample::Trades { replace: false } => {
            let mut order: Vec<usize> = (0..n).collect();
            for _ in 0..draws {
                // Fisher-Yates over the previous order: each draw is a uniform permutation
                for i in (1..n).rev() {
                    let j = rng.below(i as u64 + 1) as usize;
                    order.swap(i, j);
                }
                let (_, fall) = walk(&log_growth, order.iter().copied());
                falls.push(fall);
            }
        }
    }
    let ending_fixed = levels.is_empty();

    let years = periods_per_year.map(|ppy| n as f64 / ppy);
    let (ending_return, annual_return, prob_loss) = if ending_fixed {
        // The same observations in another order: the sum of their logs is the observed one
        // (taken once, exactly rounded), not a reordered float sum that differs in the last bit.
        let annual = years.map(|y| Percentiles::all((observed_level / y).exp_m1()));
        let loss = if observed_level < 0.0 { 1.0 } else { 0.0 };
        (Percentiles::all(observed_level.exp_m1()), annual, loss)
    } else {
        levels.sort_by(f64::total_cmp);
        let annual = years.map(|y| Percentiles::of(&levels, |l| (l / y).exp_m1()));
        let losses = levels.iter().filter(|l| **l < 0.0).count();
        (Percentiles::of(&levels, f64::exp_m1), annual, losses as f64 / draws as f64)
    };

    let mut drawdowns: Vec<f64> = falls.iter().map(|f| drawdown_of(*f)).collect();
    drawdowns.sort_by(f64::total_cmp);
    let prob_drawdown = drawdown_levels
        .iter()
        .map(|level| {
            // drawdowns are sorted: the paths at or past the level are the tail from its first
            let first = drawdowns.partition_point(|d| *d < *level);
            (*level, (draws - first) as f64 / draws as f64)
        })
        .collect();
    let top = drawdowns[draws - 1];
    let hi = if top > 0.0 { top } else { 1.0 };
    let width = hi / DRAWDOWN_BINS as f64;
    let edges: Vec<f64> =
        (0..=DRAWDOWN_BINS).map(|k| hi * k as f64 / DRAWDOWN_BINS as f64).collect();
    let mut counts = vec![0u64; DRAWDOWN_BINS];
    for d in &drawdowns {
        let bin = ((d / width) as usize).min(DRAWDOWN_BINS - 1);
        counts[bin] += 1;
    }

    Ok(Paths {
        n,
        draws,
        block,
        replace: !matches!(method, Resample::Trades { replace: false }),
        observed_return: observed_level.exp_m1(),
        observed_drawdown: drawdown_of(observed_fall),
        ending_return,
        annual_return,
        max_drawdown: Percentiles::of(&drawdowns, |d| d),
        prob_loss,
        prob_drawdown,
        edges,
        counts,
        ending_fixed,
    })
}

/// The sum of a return series with its `top` best periods left out.
pub fn without_best(returns: &[f64], top: i64) -> Result<f64> {
    if top < 0 {
        return Err(RaptorError::invalid_parameter("top is not negative"));
    }
    let mut kept = returns.to_vec();
    kept.sort_by(f64::total_cmp);
    let keep = kept.len().saturating_sub(usize::try_from(top).unwrap_or(usize::MAX));
    Ok(fsum(kept[..keep].iter().copied()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(seed: u64, n: usize, scale: f64) -> Vec<f64> {
        let mut r = SplitMix64::new(seed);
        (0..n).map(|_| (r.uniform() - 0.5) * scale).collect()
    }

    // Pinned reference values on seeded noise inputs.
    #[test]
    fn deflated_sharpe_matches_reference() {
        let sharpes = noise(3, 40, 2.0);
        let best = sharpes.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        match deflated_sharpe(&sharpes, best, 250, 0.0, 3.0, 252.0, Luck::Trials) {
            DeflatedSharpe::Computed { trials, sr0_annualised, dsr, z, .. } => {
                assert_eq!(trials, 40);
                assert!((sr0_annualised - PIN_SR0).abs() < 1e-12, "{sr0_annualised}");
                assert!((dsr - PIN_DSR).abs() < 1e-12, "{dsr}");
                assert!((z - PIN_Z).abs() < 1e-12, "{z}");
            }
            other => panic!("{other:?}"),
        }
        match deflated_sharpe(&[0.0; 64], 1.0, 1001, 0.0, 3.0, 252.0, Luck::Samples) {
            DeflatedSharpe::Computed { sr0_annualised, dsr, .. } => {
                assert!((sr0_annualised - PIN_SR0_SAMPLES).abs() < 1e-12, "{sr0_annualised}");
                assert!((dsr - PIN_DSR_SAMPLES).abs() < 1e-12, "{dsr}");
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(
            deflated_sharpe(&[1.0], 1.0, 10, 0.0, 3.0, 252.0, Luck::Trials),
            DeflatedSharpe::Insufficient { trials: 1 }
        );
        assert!(matches!(
            deflated_sharpe(&[0.0, 1.0], 100.0, 100, 5.0, 3.0, 252.0, Luck::Trials),
            DeflatedSharpe::MomentsOutOfRange { trials: 2, .. }
        ));
        assert!(Luck::parse("guess").is_err());
    }

    #[test]
    fn bootstrap_matches_reference() {
        let noisy = noise(11, 250, 0.02);
        match stationary_bootstrap(&noisy, 400, Some(5.0), 7, 252.0).unwrap() {
            Bootstrap::Drawn { p05, p50, p95, above_zero, block, .. } => {
                assert_eq!(block, 5.0);
                assert!((p05 - PIN_P05).abs() < 1e-12, "{p05}");
                assert!((p50 - PIN_P50).abs() < 1e-12, "{p50}");
                assert!((p95 - PIN_P95).abs() < 1e-12, "{p95}");
                assert_eq!(above_zero, PIN_ABOVE);
            }
            other => panic!("{other:?}"),
        }
        match stationary_bootstrap(&[0.001; 60], 50, None, 1, 252.0).unwrap() {
            Bootstrap::Drawn { block, p05, p95, above_zero, .. } => {
                assert_eq!(block, 4.0);
                assert!((p05 - 0.252).abs() < 1e-12 && (p95 - 0.252).abs() < 1e-12);
                assert_eq!(above_zero, 1.0);
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(
            stationary_bootstrap(&[0.01], 2000, None, 1, 252.0).unwrap(),
            Bootstrap::Insufficient { samples: 1 }
        );
        assert!(stationary_bootstrap(&noisy, 10, Some(0.5), 1, 252.0).is_err());
        assert!(stationary_bootstrap(&noisy, 10, Some(f64::NAN), 1, 252.0).is_err());
    }

    #[test]
    fn without_best_drops_the_luckiest_periods() {
        let xs = [1.0, 2.0, 3.0, -1.0];
        assert_eq!(without_best(&xs, 0).unwrap(), 5.0);
        assert_eq!(without_best(&xs, 1).unwrap(), 2.0);
        assert_eq!(without_best(&xs, 2).unwrap(), 0.0);
        assert_eq!(without_best(&xs, 9).unwrap(), 0.0);
        assert!(without_best(&xs, -1).is_err());
        let noisy = noise(11, 250, 0.02);
        assert!((without_best(&noisy, 5).unwrap() - PIN_WITHOUT_5).abs() < 1e-12);
    }

    const LEVELS: [f64; 3] = [0.1, 0.2, 0.3];

    #[test]
    fn resampled_paths_of_constant_returns_never_fall() {
        let flat = [0.001; 60];
        for method in [
            Resample::Block { block: None },
            Resample::Trades { replace: true },
            Resample::Trades { replace: false },
        ] {
            let p = resample_paths(&flat, method, 200, 3, Some(252.0), &LEVELS).unwrap();
            let end = 1.001f64.powi(60) - 1.0;
            for v in [p.ending_return.p05, p.ending_return.p99, p.observed_return] {
                assert!((v - end).abs() < 1e-12, "{v}");
            }
            let annual = p.annual_return.unwrap();
            assert!((annual.p50 - (1.001f64.powf(252.0) - 1.0)).abs() < 1e-9);
            assert_eq!(p.max_drawdown, Percentiles::all(0.0));
            assert_eq!(p.observed_drawdown, 0.0);
            assert_eq!(p.prob_loss, 0.0);
            assert!(p.prob_drawdown.iter().all(|(_, prob)| *prob == 0.0));
            // no drawdown anywhere: the edges run to 1 and every path is in the first bin
            assert_eq!(p.edges.len(), DRAWDOWN_BINS + 1);
            assert_eq!((p.edges[0], p.edges[DRAWDOWN_BINS]), (0.0, 1.0));
            assert_eq!(p.counts[0], 200);
        }
        let p = resample_paths(&flat, Resample::Block { block: None }, 10, 3, None, &[]).unwrap();
        assert_eq!(p.block, Some(4.0)); // the cube root of 60, rounded
        assert!(p.annual_return.is_none() && p.prob_drawdown.is_empty());
    }

    #[test]
    fn a_shuffle_keeps_the_ending_return_and_spreads_the_drawdown() {
        // The non-zero trades come in a uniform order of (+0.1, -0.1, -0.1): two losses in a
        // row (4 orders of 6) fall 1 - 0.9^2 = 0.19 from the peak; a gain between them leaves
        // the second loss 1 - 0.9 * 1.1 * 0.9 = 0.109 under the start.
        let mut trades = [0.0; 10];
        trades[2] = 0.1;
        trades[5] = -0.1;
        trades[8] = -0.1;
        let p =
            resample_paths(&trades, Resample::Trades { replace: false }, 6000, 9, None, &[0.15])
                .unwrap();
        assert!(p.ending_fixed && !p.replace);
        let end = 1.1 * 0.9 * 0.9 - 1.0;
        assert!((p.observed_return - end).abs() < 1e-14);
        assert_eq!(p.ending_return, Percentiles::all(p.observed_return));
        assert_eq!(p.prob_loss, 1.0);
        for d in [p.max_drawdown.p05, p.max_drawdown.p50, p.max_drawdown.p95, p.max_drawdown.p99] {
            assert!((d - 0.19).abs() < 1e-12 || (d - 0.109).abs() < 1e-12, "{d}");
        }
        assert!((p.prob_drawdown[0].1 - 2.0 / 3.0).abs() < 0.03, "{:?}", p.prob_drawdown);
        assert_eq!(p.counts.iter().sum::<u64>(), 6000);
        assert!((p.edges[DRAWDOWN_BINS] - 0.19).abs() < 1e-12);
        // the observed order: the gain first, then both losses
        assert!((p.observed_drawdown - 0.19).abs() < 1e-12);
    }

    #[test]
    fn resampled_paths_are_a_function_of_the_seed_and_ordered() {
        let noisy = noise(11, 250, 0.04);
        for method in [
            Resample::Block { block: Some(5.0) },
            Resample::Trades { replace: true },
            Resample::Trades { replace: false },
        ] {
            let a = resample_paths(&noisy, method, 500, 7, Some(252.0), &LEVELS).unwrap();
            assert_eq!(a, resample_paths(&noisy, method, 500, 7, Some(252.0), &LEVELS).unwrap());
            assert_ne!(a, resample_paths(&noisy, method, 500, 8, Some(252.0), &LEVELS).unwrap());
            let (e, d) = (a.ending_return, a.max_drawdown);
            assert!(e.p05 <= e.p50 && e.p50 <= e.p95 && e.p95 <= e.p99);
            assert!(d.p05 <= d.p50 && d.p50 <= d.p95 && d.p95 <= d.p99);
            assert!(0.0 <= d.p05 && d.p99 <= 1.0);
            let probs: Vec<f64> = a.prob_drawdown.iter().map(|(_, p)| *p).collect();
            assert!(probs[0] >= probs[1] && probs[1] >= probs[2]);
            assert_eq!(a.counts.iter().sum::<u64>(), 500);
        }
        let ruined = [-1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
        let p = resample_paths(&ruined, Resample::Trades { replace: false }, 50, 1, None, &[1.0])
            .unwrap();
        assert_eq!((p.observed_return, p.observed_drawdown), (-1.0, 1.0));
        assert_eq!(p.prob_drawdown, vec![(1.0, 1.0)]);
    }

    #[test]
    fn resample_paths_refuses_what_it_cannot_answer() {
        let ok = [0.01; 10];
        let run = |v: &[f64], m: Resample, draws: usize, ppy: Option<f64>, levels: &[f64]| {
            resample_paths(v, m, draws, 1, ppy, levels).unwrap_err().code()
        };
        let block = Resample::Block { block: None };
        assert_eq!(run(&ok[..9], block, 10, None, &[]), "insufficient_observations");
        let mut bad = ok;
        bad[3] = f64::NAN;
        assert_eq!(run(&bad, block, 10, None, &[]), "invalid_input");
        bad[3] = -1.5;
        assert_eq!(run(&bad, block, 10, None, &[]), "invalid_input");
        assert_eq!(run(&ok, block, 0, None, &[]), "invalid_input");
        assert_eq!(run(&ok, block, RESAMPLE_MAX_DRAWS + 1, None, &[]), "invalid_input");
        assert_eq!(run(&ok, block, 10, None, &[0.0]), "invalid_input");
        assert_eq!(run(&ok, block, 10, None, &[1.5]), "invalid_input");
        assert_eq!(run(&ok, block, 10, None, &[0.1; RESAMPLE_MAX_LEVELS + 1]), "invalid_input");
        assert_eq!(run(&ok, block, 10, Some(0.0), &[]), "invalid_input");
        assert_eq!(run(&ok, Resample::Block { block: Some(0.5) }, 10, None, &[]), "invalid_input");
        assert_eq!(
            run(&ok, Resample::Block { block: Some(f64::NAN) }, 10, None, &[]),
            "invalid_input"
        );
    }

    const PIN_SR0: f64 = 1.1990460870938555;
    const PIN_DSR: f64 = 0.3724223740155374;
    const PIN_Z: f64 = -0.32544441111941086;
    const PIN_SR0_SAMPLES: f64 = 1.1894233513320038;
    const PIN_DSR_SAMPLES: f64 = 0.3530992285646916;
    const PIN_P05: f64 = 0.048982786023643955;
    const PIN_P50: f64 = 0.2101849112429047;
    const PIN_P95: f64 = 0.35705171778217754;
    const PIN_ABOVE: f64 = 0.9875;
    const PIN_WITHOUT_5: f64 = 0.15654936181198048;
}
