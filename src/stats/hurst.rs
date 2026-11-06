//! Does a series remember: the Hurst exponent of its increments by rescaled range (R/S) and by
//! detrended fluctuation analysis (DFA). Above 0.5 the increments persist (trend), below 0.5
//! they reverse (mean reversion), 0.5 is a random walk.
//!
//! The scales are `unique(floor(logspace(log10(16), log10(n/4), 12)))`, evaluated as numpy
//! does: `10 ** (i * step + start)` with the last exponent set to `stop`. A scale sits on an
//! edge when `10 ** log10(m)` lands a rounding error below `m` (`10 ** log10(64)` is
//! 63.99999999999999 and floors to 63), so the evaluation order here is part of the contract.
//!
//! The estimate carries no verdict. The Python function `reading` is "persistent" above 0.55, "anti-persistent"
//! below 0.45 and "random walk" between, on the mean of the two estimates.

use crate::math::fsum::fsum;
use crate::math::special::ln_gamma;

use super::error::StatsError;
use super::hac::{finite_series, mean, peak_to_peak};

pub const MIN_OBS: usize = 256;
pub const MIN_WINDOW: usize = 16;
pub const N_SCALES: usize = 12;
/// The gamma-function form of E[R/S] up to here; Peters' approximation above.
pub const ANIS_LLOYD_EXACT_MAX: usize = 340;
pub const MIN_FIT_POINTS: usize = 4;

#[derive(Debug, Clone, PartialEq)]
pub struct RescaledRange {
    /// The raw slope with the slope an iid series shows at the same scales removed, plus 0.5.
    pub hurst: f64,
    pub raw_slope: f64,
    pub std_error: f64,
    pub scales: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Dfa {
    pub hurst: f64,
    pub std_error: f64,
    pub scales: Vec<usize>,
}

/// The result of [`hurst`], full precision.
#[derive(Debug, Clone, PartialEq)]
pub struct Hurst {
    pub n: usize,
    pub rescaled_range: RescaledRange,
    pub dfa: Dfa,
}

/// The block sizes for `n` increments: at least four blocks at the largest.
pub fn scales(n: usize) -> Vec<usize> {
    let top = (n / 4) as f64;
    let (start, stop) = ((MIN_WINDOW as f64).log10(), top.log10());
    let step = (stop - start) / (N_SCALES - 1) as f64;
    let mut out: Vec<usize> = (0..N_SCALES)
        .map(|i| {
            let exponent = if i == N_SCALES - 1 { stop } else { i as f64 * step + start };
            10.0f64.powf(exponent).floor() as usize
        })
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

fn rs_of_block(block: &[f64]) -> f64 {
    let m = mean(block);
    let s = (block.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / block.len() as f64).sqrt();
    if s <= 0.0 {
        return f64::NAN;
    }
    let (mut cum, mut hi, mut lo) = (0.0, f64::NEG_INFINITY, f64::INFINITY);
    for x in block {
        cum += x - m;
        hi = hi.max(cum);
        lo = lo.min(cum);
    }
    (hi - lo) / s
}

/// Anis and Lloyd (1976) with Peters' correction: E[R/S] of `n` iid observations.
pub fn expected_rs(n: usize) -> Result<f64, StatsError> {
    let nf = n as f64;
    let tail = fsum((1..n).map(|i| ((nf - i as f64) / i as f64).sqrt()));
    if n <= ANIS_LLOYD_EXACT_MAX {
        let gamma_form =
            (ln_gamma((nf - 1.0) / 2.0)? - ln_gamma(nf / 2.0)?).exp() / std::f64::consts::PI.sqrt();
        Ok((nf - 0.5) / nf * gamma_form * tail)
    } else {
        Ok((nf - 0.5) / nf * (1.0 / (nf * std::f64::consts::PI / 2.0).sqrt()) * tail)
    }
}

/// Slope of `y` on `x` and its standard error.
fn slope(x: &[f64], y: &[f64]) -> (f64, f64) {
    let (mx, my) = (mean(x), mean(y));
    let xc: Vec<f64> = x.iter().map(|v| v - mx).collect();
    let sxx: f64 = xc.iter().map(|v| v * v).sum();
    let b = xc.iter().zip(y).map(|(a, v)| a * (v - my)).sum::<f64>() / sxx;
    let rss: f64 = xc.iter().zip(y).map(|(a, v)| (v - my - b * a) * (v - my - b * a)).sum();
    (b, (rss / (x.len() - 2) as f64 / sxx).sqrt())
}

fn rs_hurst(inc: &[f64], scales: &[usize]) -> Result<RescaledRange, StatsError> {
    let (mut mean_rs, mut exp_rs, mut used) = (Vec::new(), Vec::new(), Vec::new());
    for &s in scales {
        let vals: Vec<f64> = inc[..inc.len() / s * s]
            .chunks_exact(s)
            .map(rs_of_block)
            .filter(|v| v.is_finite())
            .collect();
        if !vals.is_empty() {
            mean_rs.push(mean(&vals));
            exp_rs.push(expected_rs(s)?);
            used.push(s);
        }
    }
    if used.len() < MIN_FIT_POINTS {
        return Err(StatsError::DegenerateSeries("too few usable scales for R/S".into()));
    }
    let lx: Vec<f64> = used.iter().map(|&s| (s as f64).ln()).collect();
    let ly: Vec<f64> = mean_rs.iter().map(|v| v.ln()).collect();
    let le: Vec<f64> = exp_rs.iter().map(|v| v.ln()).collect();
    let (raw, se) = slope(&lx, &ly);
    let (null, _) = slope(&lx, &le);
    Ok(RescaledRange { hurst: raw - null + 0.5, raw_slope: raw, std_error: se, scales: used })
}

fn dfa_hurst(inc: &[f64], scales: &[usize]) -> Result<Dfa, StatsError> {
    let m = mean(inc);
    let mut profile = Vec::with_capacity(inc.len());
    let mut cum = 0.0;
    for x in inc {
        cum += x - m;
        profile.push(cum);
    }
    let mut fluctuation = Vec::with_capacity(scales.len());
    for &s in scales {
        let tm = (s as f64 - 1.0) / 2.0;
        let stt: f64 = (0..s).map(|t| (t as f64 - tm) * (t as f64 - tm)).sum();
        let mut rms = Vec::new();
        for block in profile[..profile.len() / s * s].chunks_exact(s) {
            let bm = mean(block);
            let b1 = block.iter().enumerate().map(|(t, v)| (t as f64 - tm) * (v - bm)).sum::<f64>()
                / stt;
            let sse: f64 = block
                .iter()
                .enumerate()
                .map(|(t, v)| {
                    let r = v - bm - b1 * (t as f64 - tm);
                    r * r
                })
                .sum();
            rms.push(sse / s as f64);
        }
        fluctuation.push(mean(&rms).sqrt());
    }
    if fluctuation.iter().any(|f| f.is_nan() || *f <= 0.0) {
        return Err(StatsError::DegenerateSeries(
            "the detrended fluctuation is zero at some scale".into(),
        ));
    }
    let lx: Vec<f64> = scales.iter().map(|&s| (s as f64).ln()).collect();
    let ly: Vec<f64> = fluctuation.iter().map(|v| v.ln()).collect();
    let (alpha, se) = slope(&lx, &ly);
    Ok(Dfa { hurst: alpha, std_error: se, scales: scales.to_vec() })
}

/// `increments` are returns (log price changes), not prices.
pub fn hurst(increments: &[f64]) -> Result<Hurst, StatsError> {
    let a = finite_series(increments, "increments", MIN_OBS)?;
    if peak_to_peak(a) == 0.0 {
        return Err(StatsError::DegenerateSeries("increments are constant".into()));
    }
    let sc = scales(a.len());
    if sc.len() < MIN_FIT_POINTS {
        return Err(StatsError::DegenerateSeries("too few scales for this length".into()));
    }
    Ok(Hurst { n: a.len(), rescaled_range: rs_hurst(a, &sc)?, dfa: dfa_hurst(a, &sc)? })
}
