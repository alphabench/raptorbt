//! The deflated Sharpe ratio, the stationary bootstrap and `without_best`.

use crate::core::error::{RaptorError, Result};
use crate::math::fsum::fsum;
use crate::pricing::norm::{norm_cdf, norm_ppf};
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
    let mean_block = match block {
        Some(b) => b,
        None => (n as f64).powf(1.0 / 3.0).round_ties_even().max(1.0),
    };
    // NaN is refused too (it would otherwise run as "never start a new block")
    if mean_block.is_nan() || mean_block < 1.0 {
        return Err(RaptorError::invalid_parameter("block is at least one period"));
    }
    let p_new = 1.0 / mean_block;
    let mut rng = SplitMix64::new(seed);
    let len = n as u64;
    let mut means = Vec::with_capacity(draws);
    for _ in 0..draws {
        let mut i = rng.below(len) as usize;
        let mut total = 0.0;
        for _ in 0..n {
            total += returns[i];
            i = if rng.uniform() < p_new { rng.below(len) as usize } else { (i + 1) % n };
        }
        means.push(total / n as f64 * periods_per_year);
    }
    means.sort_by(f64::total_cmp);
    let pct = |q: f64| -> f64 {
        let k = (q * draws as f64).ceil() as i64 - 1;
        means[k.clamp(0, draws as i64 - 1) as usize]
    };
    let above = means.iter().filter(|m| **m > 0.0).count();
    Ok(Bootstrap::Drawn {
        samples: n,
        draws,
        block: mean_block,
        p05: pct(0.05),
        p50: pct(0.50),
        p95: pct(0.95),
        above_zero: above as f64 / draws as f64,
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
