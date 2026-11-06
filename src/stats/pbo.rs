//! The probability of backtest overfitting by combinatorially symmetric cross-validation
//! (Bailey, Borwein, Lopez de Prado and Zhu): the share of ways to choose half of the time
//! blocks as the in-sample half for which the in-sample best strategy ranks at or below the
//! median out of sample.
//!
//! The combinations are scored in parallel and collected in their lexicographic order, then
//! reduced in that order with correctly rounded sums, so the answer does not depend on the
//! number of threads.

use rayon::prelude::*;

use crate::math::fsum::fsum;

use super::error::StatsError;
use super::hac::finite_series;

pub const MIN_SPLITS: usize = 4;
pub const MAX_STRATEGIES: usize = 200;
pub const MAX_COMBINATIONS: u128 = 20_000;

/// The result of [`pbo_cscv`], full precision. `dropped_rows` is how many of the oldest rows were
/// left out to make the blocks equal.
#[derive(Debug, Clone, PartialEq)]
pub struct Pbo {
    pub n_periods: usize,
    pub n_strategies: usize,
    pub splits: usize,
    pub combinations: usize,
    pub dropped_rows: usize,
    pub pbo: f64,
    pub median_logit: f64,
    pub mean_in_sample_sharpe_of_best: f64,
    pub mean_out_of_sample_sharpe_of_best: f64,
    pub prob_out_of_sample_loss_of_best: f64,
}

/// `C(n, k)`, saturating at `u128::MAX`.
fn choose(n: usize, k: usize) -> u128 {
    let k = k.min(n - k);
    let mut r: u128 = 1;
    for i in 0..k {
        match r.checked_mul((n - i) as u128) {
            Some(v) => r = v / (i as u128 + 1),
            None => return u128::MAX,
        }
    }
    r
}

/// Every `choose`-subset of `0..n` as a bit mask, in lexicographic order of the indices.
fn subsets(n: usize, choose: usize) -> Vec<u32> {
    fn go(start: usize, n: usize, left: usize, mask: u32, out: &mut Vec<u32>) {
        if left == 0 {
            out.push(mask);
            return;
        }
        for i in start..=(n - left) {
            go(i + 1, n, left - 1, mask | (1 << i), out);
        }
    }
    let mut out = Vec::new();
    go(0, n, choose, 0, &mut out);
    out
}

/// Per-block sums of the returns and of their squares: `[block][strategy]`.
struct BlockSums {
    s: Vec<Vec<f64>>,
    ss: Vec<Vec<f64>>,
    cnt: usize,
}

impl BlockSums {
    /// Sharpe ratio (per period) of each strategy over the blocks in `mask`, zero where the
    /// variance is not positive.
    fn sharpe(&self, mask: u32, splits: usize, out: &mut [f64]) {
        let blocks = (0..splits).filter(|b| mask >> b & 1 == 1);
        let n = (self.cnt * mask.count_ones() as usize) as f64;
        for (j, o) in out.iter_mut().enumerate() {
            let (mut s, mut ss) = (0.0, 0.0);
            for b in blocks.clone() {
                s += self.s[b][j];
                ss += self.ss[b][j];
            }
            let mean = s / n;
            let var = (ss - n * (mean * mean)) / (n - 1.0);
            *o = if var > 0.0 { mean / var.sqrt() } else { 0.0 };
        }
    }
}

/// Rows are periods, one column per strategy. The oldest rows beyond a whole number of blocks
/// are dropped. `splits` is even and at least 4.
pub fn pbo_cscv<C: AsRef<[f64]>>(cols: &[C], splits: usize) -> Result<Pbo, StatsError> {
    if splits < MIN_SPLITS || splits % 2 == 1 {
        return Err(StatsError::Invalid(format!(
            "splits must be an even number, at least {MIN_SPLITS}"
        )));
    }
    let min_rows = splits * 4;
    if cols.len() < 2 {
        return Err(StatsError::InsufficientObservations {
            what: "strategies".into(),
            found: cols.len(),
            needed: 2,
        });
    }
    let series = cols
        .iter()
        .enumerate()
        .map(|(i, c)| finite_series(c.as_ref(), &format!("strategies[{i}]"), min_rows))
        .collect::<Result<Vec<_>, _>>()?;
    let len = series[0].len();
    if series.iter().any(|s| s.len() != len) {
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
    let n_comb = choose(splits, splits / 2);
    if n_comb > MAX_COMBINATIONS {
        return Err(StatsError::Invalid(format!(
            "splits={splits} makes {n_comb} combinations, at most {MAX_COMBINATIONS}"
        )));
    }

    let cnt = len / splits;
    let rows = cnt * splits;
    let dropped_rows = len - rows;
    let mut sums = BlockSums { s: Vec::with_capacity(splits), ss: Vec::with_capacity(splits), cnt };
    for b in 0..splits {
        let rng = dropped_rows + b * cnt..dropped_rows + (b + 1) * cnt;
        let mut s = Vec::with_capacity(k);
        let mut ss = Vec::with_capacity(k);
        for col in &series {
            let (mut a, mut q) = (0.0, 0.0);
            for &x in &col[rng.clone()] {
                a += x;
                q += x * x;
            }
            s.push(a);
            ss.push(q);
        }
        sums.s.push(s);
        sums.ss.push(ss);
    }

    let all: u32 = (1u32 << splits) - 1;
    let masks = subsets(splits, splits / 2);
    // (logit, in-sample Sharpe of the best, its out-of-sample Sharpe), one per combination
    let rows_out: Vec<(f64, f64, f64)> = masks
        .par_iter()
        .map(|&ins| {
            let mut sr_is = vec![0.0; k];
            let mut sr_oos = vec![0.0; k];
            sums.sharpe(ins, splits, &mut sr_is);
            sums.sharpe(all ^ ins, splits, &mut sr_oos);
            let mut best = 0;
            for (j, v) in sr_is.iter().enumerate() {
                if *v > sr_is[best] {
                    best = j;
                }
            }
            let target = sr_oos[best];
            let less = sr_oos.iter().filter(|v| **v < target).count();
            let equal = sr_oos.iter().filter(|v| **v == target).count();
            // 1 = worst out of sample; ties share
            let rank = 1.0 + less as f64 + 0.5 * (equal as f64 - 1.0);
            let w = rank / (k as f64 + 1.0);
            ((w / (1.0 - w)).ln(), sr_is[best], target)
        })
        .collect();

    let nc = rows_out.len() as f64;
    let mut logits: Vec<f64> = rows_out.iter().map(|r| r.0).collect();
    let pbo = logits.iter().filter(|l| **l <= 0.0).count() as f64 / nc;
    logits.sort_by(f64::total_cmp);
    let mid = logits.len() / 2;
    let median_logit =
        if logits.len() % 2 == 1 { logits[mid] } else { (logits[mid - 1] + logits[mid]) / 2.0 };
    Ok(Pbo {
        n_periods: rows,
        n_strategies: k,
        splits,
        combinations: rows_out.len(),
        dropped_rows,
        pbo,
        median_logit,
        mean_in_sample_sharpe_of_best: fsum(rows_out.iter().map(|r| r.1)) / nc,
        mean_out_of_sample_sharpe_of_best: fsum(rows_out.iter().map(|r| r.2)) / nc,
        prob_out_of_sample_loss_of_best: rows_out.iter().filter(|r| r.2 < 0.0).count() as f64 / nc,
    })
}
