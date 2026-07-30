//! Cross-sectional factor scoring over a panel.
//!
//! All panels are row-major `n_dates x n_assets`. NaN means "asset absent on
//! that date" and is propagated, never imputed. A date whose cross-section has
//! fewer than `min_names` finite values produces an all-NaN output row -- the
//! caller decides whether that day is usable. Infinities are always a hard
//! error: absence is representable (NaN), a broken input is not.
//!
//! Factor *lists* and composite *weights* are caller-supplied data. This
//! module hardcodes no factor and no weighting -- the platform's factor
//! configuration (and each factor's measured IC provenance) lives in the
//! backend, not in Rust.

use super::errors::PortfolioMathError;

fn check_shape(values: &[f64], n_dates: usize, n_assets: usize) -> Result<(), PortfolioMathError> {
    if values.len() != n_dates * n_assets {
        return Err(PortfolioMathError::ShapeMismatch(format!(
            "expected {n_dates}x{n_assets}={} values, got {}",
            n_dates * n_assets,
            values.len()
        )));
    }
    for (idx, v) in values.iter().enumerate() {
        if v.is_infinite() {
            return Err(PortfolioMathError::NonFinite { row: idx / n_assets, col: idx % n_assets });
        }
    }
    Ok(())
}

/// Indices of finite values in one date row.
fn finite_cols(row: &[f64]) -> Vec<usize> {
    row.iter().enumerate().filter(|(_, v)| v.is_finite()).map(|(i, _)| i).collect()
}

/// Winsorize each date's cross-section at the `pct`/`1-pct` quantiles.
pub fn winsorize_panel(
    values: &[f64],
    n_dates: usize,
    n_assets: usize,
    pct: f64,
) -> Result<Vec<f64>, PortfolioMathError> {
    check_shape(values, n_dates, n_assets)?;
    if !(0.0..0.5).contains(&pct) {
        return Err(PortfolioMathError::DegenerateInput(format!(
            "winsorize pct must be in [0, 0.5), got {pct}"
        )));
    }
    let mut out = values.to_vec();
    for d in 0..n_dates {
        let row = &values[d * n_assets..(d + 1) * n_assets];
        let cols = finite_cols(row);
        if cols.len() < 2 {
            continue;
        }
        let mut sorted: Vec<f64> = cols.iter().map(|&c| row[c]).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Symmetric count-based winsorization: clip the k lowest and k
        // highest, k = floor(m * pct). On a cross-section too small for pct
        // to cover one name (k = 0), nothing clips -- by design.
        let k = (sorted.len() as f64 * pct).floor() as usize;
        let lo = sorted[k];
        let hi = sorted[sorted.len() - 1 - k];
        for &c in &cols {
            out[d * n_assets + c] = row[c].clamp(lo, hi);
        }
    }
    Ok(out)
}

/// Z-score each date's cross-section: (x - mean) / std.
///
/// Dates with fewer than `min_names` finite values, or with zero
/// cross-sectional dispersion, produce an all-NaN row.
pub fn zscore_panel(
    values: &[f64],
    n_dates: usize,
    n_assets: usize,
    min_names: usize,
) -> Result<Vec<f64>, PortfolioMathError> {
    check_shape(values, n_dates, n_assets)?;
    if min_names < 2 {
        return Err(PortfolioMathError::DegenerateInput(format!(
            "min_names must be >= 2, got {min_names}"
        )));
    }
    let mut out = vec![f64::NAN; n_dates * n_assets];
    for d in 0..n_dates {
        let row = &values[d * n_assets..(d + 1) * n_assets];
        let cols = finite_cols(row);
        if cols.len() < min_names {
            continue;
        }
        let m = cols.len() as f64;
        let mean: f64 = cols.iter().map(|&c| row[c]).sum::<f64>() / m;
        let var: f64 = cols.iter().map(|&c| (row[c] - mean).powi(2)).sum::<f64>() / m;
        if var <= 0.0 {
            continue; // degenerate cross-section stays NaN
        }
        let std = var.sqrt();
        for &c in &cols {
            out[d * n_assets + c] = (row[c] - mean) / std;
        }
    }
    Ok(out)
}

/// Rank each date's cross-section into [0, 1], ties averaged.
pub fn rank_panel(
    values: &[f64],
    n_dates: usize,
    n_assets: usize,
    min_names: usize,
) -> Result<Vec<f64>, PortfolioMathError> {
    check_shape(values, n_dates, n_assets)?;
    if min_names < 2 {
        return Err(PortfolioMathError::DegenerateInput(format!(
            "min_names must be >= 2, got {min_names}"
        )));
    }
    let mut out = vec![f64::NAN; n_dates * n_assets];
    for d in 0..n_dates {
        let row = &values[d * n_assets..(d + 1) * n_assets];
        let cols = finite_cols(row);
        if cols.len() < min_names {
            continue;
        }
        let mut order: Vec<usize> = cols.clone();
        order.sort_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap());
        let m = order.len();
        // Average-rank ties.
        let mut i = 0;
        while i < m {
            let mut j = i;
            while j + 1 < m && row[order[j + 1]] == row[order[i]] {
                j += 1;
            }
            let avg_rank = (i + j) as f64 / 2.0;
            let denom = (m - 1) as f64;
            for &col in &order[i..=j] {
                out[d * n_assets + col] = if denom > 0.0 { avg_rank / denom } else { 0.5 };
            }
            i = j + 1;
        }
    }
    Ok(out)
}

/// Price momentum with a skip window: p[t-skip] / p[t-lookback] - 1.
///
/// The classic 12-1 on daily bars is `lookback=252, skip=21`. Output is NaN
/// until `lookback` observations exist for the asset, and NaN whenever either
/// endpoint price is NaN or non-positive.
pub fn momentum_panel(
    prices: &[f64],
    n_dates: usize,
    n_assets: usize,
    lookback: usize,
    skip: usize,
) -> Result<Vec<f64>, PortfolioMathError> {
    check_shape(prices, n_dates, n_assets)?;
    if lookback == 0 || skip >= lookback {
        return Err(PortfolioMathError::DegenerateInput(format!(
            "need 0 <= skip < lookback, got lookback={lookback} skip={skip}"
        )));
    }
    let mut out = vec![f64::NAN; n_dates * n_assets];
    for d in lookback..n_dates {
        for a in 0..n_assets {
            let p_new = prices[(d - skip) * n_assets + a];
            let p_old = prices[(d - lookback) * n_assets + a];
            if p_new.is_finite() && p_old.is_finite() && p_old > 0.0 && p_new > 0.0 {
                out[d * n_assets + a] = p_new / p_old - 1.0;
            }
        }
    }
    Ok(out)
}

/// Weighted composite of factor panels.
///
/// `factors` are same-shaped panels; `weights` must be finite with a positive
/// sum (they are renormalized to sum to 1). An asset-date is NaN in the output
/// if it is NaN in ANY input factor -- a composite over partial information
/// would silently favor data-sparse names.
pub fn composite_scores(
    factors: &[&[f64]],
    weights: &[f64],
    n_dates: usize,
    n_assets: usize,
) -> Result<Vec<f64>, PortfolioMathError> {
    if factors.is_empty() {
        return Err(PortfolioMathError::DegenerateInput("no factor panels supplied".into()));
    }
    if weights.len() != factors.len() {
        return Err(PortfolioMathError::ShapeMismatch(format!(
            "{} weights for {} factors",
            weights.len(),
            factors.len()
        )));
    }
    for f in factors {
        check_shape(f, n_dates, n_assets)?;
    }
    let w_sum: f64 = weights.iter().sum();
    if !w_sum.is_finite() || w_sum <= 0.0 || weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
        return Err(PortfolioMathError::DegenerateInput(format!(
            "factor weights must be non-negative and sum to a positive number, got {weights:?}"
        )));
    }
    let norm: Vec<f64> = weights.iter().map(|w| w / w_sum).collect();
    let mut out = vec![f64::NAN; n_dates * n_assets];
    for idx in 0..n_dates * n_assets {
        let mut acc = 0.0;
        let mut ok = true;
        for (f, w) in factors.iter().zip(norm.iter()) {
            let v = f[idx];
            if v.is_nan() {
                ok = false;
                break;
            }
            acc += w * v;
        }
        if ok {
            out[idx] = acc;
        }
    }
    Ok(out)
}

/// Rank information coefficient of a factor against forward returns.
///
/// The per-date Spearman correlation between the factor's cross-sectional rank
/// and the rank of the return realized over the NEXT `horizon` rows, then the
/// mean of those daily ICs with a Newey-West-free t-statistic
/// `mean / (stdev / sqrt(n))` over the dates that scored.
///
/// This is the falsifiability tool: a factor earns a place in the platform only
/// by clearing a t-bar measured here, and the number it returns is the number
/// that must be persisted as that factor's provenance.
///
/// Deliberate choices, each of which changes the answer:
/// - **Forward returns are computed from `prices`, not supplied.** A caller
///   passing its own forward returns can silently leak lookahead; the shift is
///   done here where `horizon` is unambiguous.
/// - **Ranks, not levels.** Momentum's cross-section is fat-tailed, so a
///   Pearson IC on raw values is dominated by a handful of names.
/// - **A date is scored only if at least `min_names` assets have BOTH a finite
///   factor value and a finite forward return.** Pairing after the intersection
///   is what stops a thin tail of the panel producing a confident-looking IC.
/// - **NaN dates are skipped, never zero-filled.** A zero IC is a measurement;
///   an absent one is not.
pub struct RankIc {
    /// Mean of the per-date rank correlations.
    pub mean_ic: f64,
    /// Sample standard deviation of the per-date rank correlations.
    pub stdev_ic: f64,
    /// `mean_ic / (stdev_ic / sqrt(n_dates_scored))`; 0 when undefined.
    pub t_stat: f64,
    /// Number of dates that produced an IC.
    pub n_dates_scored: usize,
    /// Mean number of paired names across the scored dates.
    pub mean_names: f64,
    /// The per-date ICs, NaN on dates that did not score.
    pub daily_ic: Vec<f64>,
}

/// Spearman correlation of two equal-length finite slices, ties averaged.
fn spearman(a: &[f64], b: &[f64]) -> Option<f64> {
    let n = a.len();
    if n < 2 {
        return None;
    }
    let rank = |v: &[f64]| -> Vec<f64> {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap());
        let mut r = vec![0.0; n];
        let mut i = 0;
        while i < n {
            let mut j = i;
            while j + 1 < n && v[order[j + 1]] == v[order[i]] {
                j += 1;
            }
            let avg = (i + j) as f64 / 2.0;
            for &idx in &order[i..=j] {
                r[idx] = avg;
            }
            i = j + 1;
        }
        r
    };
    let ra = rank(a);
    let rb = rank(b);
    let mean_a: f64 = ra.iter().sum::<f64>() / n as f64;
    let mean_b: f64 = rb.iter().sum::<f64>() / n as f64;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for i in 0..n {
        let da = ra[i] - mean_a;
        let db = rb[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    // A constant cross-section on either side has no correlation to report.
    if var_a <= 0.0 || var_b <= 0.0 {
        return None;
    }
    Some(cov / (var_a * var_b).sqrt())
}

/// Measure a factor panel's rank IC against `horizon`-ahead returns.
pub fn rank_ic(
    factor: &[f64],
    prices: &[f64],
    n_dates: usize,
    n_assets: usize,
    horizon: usize,
    min_names: usize,
) -> Result<RankIc, PortfolioMathError> {
    check_shape(factor, n_dates, n_assets)?;
    check_shape(prices, n_dates, n_assets)?;
    if horizon == 0 {
        return Err(PortfolioMathError::DegenerateInput("horizon must be >= 1".into()));
    }
    if min_names < 2 {
        return Err(PortfolioMathError::DegenerateInput(format!(
            "min_names must be >= 2, got {min_names}"
        )));
    }
    let mut daily_ic = vec![f64::NAN; n_dates];
    let mut ics: Vec<f64> = Vec::new();
    let mut names_total = 0usize;
    // A date can only score if its forward window lands inside the panel.
    for d in 0..n_dates.saturating_sub(horizon) {
        let mut xs: Vec<f64> = Vec::new();
        let mut ys: Vec<f64> = Vec::new();
        for a in 0..n_assets {
            let f = factor[d * n_assets + a];
            let p0 = prices[d * n_assets + a];
            let p1 = prices[(d + horizon) * n_assets + a];
            if f.is_finite() && p0.is_finite() && p1.is_finite() && p0 > 0.0 && p1 > 0.0 {
                xs.push(f);
                ys.push(p1 / p0 - 1.0);
            }
        }
        if xs.len() < min_names {
            continue;
        }
        if let Some(ic) = spearman(&xs, &ys) {
            daily_ic[d] = ic;
            ics.push(ic);
            names_total += xs.len();
        }
    }
    let n = ics.len();
    if n == 0 {
        return Ok(RankIc {
            mean_ic: f64::NAN,
            stdev_ic: f64::NAN,
            t_stat: f64::NAN,
            n_dates_scored: 0,
            mean_names: f64::NAN,
            daily_ic,
        });
    }
    let mean_ic: f64 = ics.iter().sum::<f64>() / n as f64;
    // Sample stdev (n-1); a single date has no dispersion to report.
    let stdev_ic = if n > 1 {
        (ics.iter().map(|v| (v - mean_ic) * (v - mean_ic)).sum::<f64>() / (n - 1) as f64).sqrt()
    } else {
        f64::NAN
    };
    let t_stat = if stdev_ic.is_finite() && stdev_ic > 0.0 {
        mean_ic / (stdev_ic / (n as f64).sqrt())
    } else {
        f64::NAN
    };
    Ok(RankIc {
        mean_ic,
        stdev_ic,
        t_stat,
        n_dates_scored: n,
        mean_names: names_total as f64 / n as f64,
        daily_ic,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_ic_is_one_when_factor_perfectly_orders_forward_returns() {
        // 3 assets, factor == next-period return order on every date.
        let n_dates = 5;
        let n_assets = 3;
        // Prices rise fastest for asset 2, slowest for asset 0.
        let mut prices = vec![0.0; n_dates * n_assets];
        for d in 0..n_dates {
            for a in 0..n_assets {
                prices[d * n_assets + a] = (1.0 + 0.01 * (a as f64 + 1.0)).powi(d as i32);
            }
        }
        // Factor ranks assets in the same order as their growth rate.
        let factor: Vec<f64> = (0..n_dates).flat_map(|_| (0..n_assets).map(|a| a as f64)).collect();
        let out = rank_ic(&factor, &prices, n_dates, n_assets, 1, 2).unwrap();
        assert_eq!(out.n_dates_scored, 4); // last date has no forward window
        assert!((out.mean_ic - 1.0).abs() < 1e-12);
        assert!((out.mean_names - 3.0).abs() < 1e-12);
    }

    #[test]
    fn rank_ic_is_minus_one_when_factor_inverts_forward_returns() {
        let n_dates = 4;
        let n_assets = 3;
        let mut prices = vec![0.0; n_dates * n_assets];
        for d in 0..n_dates {
            for a in 0..n_assets {
                prices[d * n_assets + a] = (1.0 + 0.01 * (a as f64 + 1.0)).powi(d as i32);
            }
        }
        // Reversed factor: highest score on the worst performer.
        let factor: Vec<f64> =
            (0..n_dates).flat_map(|_| (0..n_assets).map(|a| -(a as f64))).collect();
        let out = rank_ic(&factor, &prices, n_dates, n_assets, 1, 2).unwrap();
        assert!((out.mean_ic + 1.0).abs() < 1e-12);
    }

    #[test]
    fn rank_ic_skips_dates_below_min_names_and_never_zero_fills() {
        let n_assets = 3;
        let n_dates = 3;
        // Date 0 has 3 finite factor values, dates 1-2 have only 1.
        let factor = vec![
            0.0,
            1.0,
            2.0, //
            0.0,
            f64::NAN,
            f64::NAN, //
            0.0,
            f64::NAN,
            f64::NAN,
        ];
        let prices = vec![100.0; n_dates * n_assets];
        // Constant prices -> forward returns all zero -> no dispersion, so even
        // date 0 cannot report a correlation.
        let out = rank_ic(&factor, &prices, n_dates, n_assets, 1, 2).unwrap();
        assert_eq!(out.n_dates_scored, 0);
        assert!(out.mean_ic.is_nan());
        assert!(out.daily_ic.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn rank_ic_refuses_bad_windows() {
        let f = vec![1.0, 2.0];
        let p = vec![1.0, 2.0];
        assert!(rank_ic(&f, &p, 1, 2, 0, 2).is_err()); // horizon 0
        assert!(rank_ic(&f, &p, 1, 2, 1, 1).is_err()); // min_names < 2
    }

    #[test]
    fn momentum_matches_identity_on_ramp() {
        // Geometric ramp: p[t] = 1.01^t. momentum = p[t-s]/p[t-l] - 1.
        let n_dates = 300;
        let prices: Vec<f64> = (0..n_dates).map(|t| 1.01f64.powi(t as i32)).collect();
        let out = momentum_panel(&prices, n_dates, 1, 252, 21).unwrap();
        for d in 252..n_dates {
            let expect = 1.01f64.powi((d - 21) as i32) / 1.01f64.powi((d - 252) as i32) - 1.0;
            assert!((out[d] - expect).abs() < 1e-10);
        }
        assert!(out[251].is_nan());
    }

    #[test]
    fn zscore_row_mean_zero_std_one() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let out = zscore_panel(&values, 1, 5, 2).unwrap();
        let mean: f64 = out.iter().sum::<f64>() / 5.0;
        let var: f64 = out.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / 5.0;
        assert!(mean.abs() < 1e-12);
        assert!((var - 1.0).abs() < 1e-12);
    }

    #[test]
    fn winsorize_pins_outlier() {
        // 11 names at 10% -> k = 1: clip one from each tail.
        let mut values: Vec<f64> = (1..=10).map(|v| v as f64).collect();
        values.push(1000.0);
        let out = winsorize_panel(&values, 1, 11, 0.10).unwrap();
        assert_eq!(out[10], 10.0); // outlier pinned to next-highest
        assert_eq!(out[0], 2.0); // low tail pinned symmetrically
        assert_eq!(out[4], 5.0); // interior untouched
    }

    #[test]
    fn winsorize_too_small_cross_section_is_a_noop() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 1000.0];
        let out = winsorize_panel(&values, 1, 5, 0.10).unwrap();
        assert_eq!(out, values); // k = floor(5*0.1) = 0: nothing to clip
    }

    #[test]
    fn rank_ties_averaged_in_unit_interval() {
        let values = vec![3.0, 1.0, 3.0, 2.0];
        let out = rank_panel(&values, 1, 4, 2).unwrap();
        assert!((out[1] - 0.0).abs() < 1e-12); // lowest
        assert!((out[3] - 1.0 / 3.0).abs() < 1e-12);
        // tied top two share (2+3)/2 / 3 = 5/6
        assert!((out[0] - 5.0 / 6.0).abs() < 1e-12);
        assert!((out[2] - 5.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn nan_column_propagates_and_min_names_gates() {
        let values = vec![1.0, f64::NAN, 3.0, 2.0, f64::NAN, f64::NAN];
        // Row 0 has 2 finite -> scored; row 1 has 1 finite -> all NaN.
        let out = zscore_panel(&values, 2, 3, 2).unwrap();
        assert!(out[1].is_nan());
        assert!(!out[0].is_nan());
        assert!(out[3].is_nan() && out[4].is_nan() && out[5].is_nan());
    }

    #[test]
    fn infinity_is_a_hard_error_nan_is_not() {
        let values = vec![1.0, f64::INFINITY, 3.0];
        assert!(matches!(
            zscore_panel(&values, 1, 3, 2),
            Err(PortfolioMathError::NonFinite { row: 0, col: 1 })
        ));
    }

    #[test]
    fn composite_requires_all_factors_present() {
        let f1 = vec![1.0, 2.0];
        let f2 = vec![f64::NAN, 4.0];
        let out = composite_scores(&[&f1, &f2], &[0.5, 0.5], 1, 2).unwrap();
        assert!(out[0].is_nan()); // missing in f2
        assert!((out[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn composite_refuses_bad_weights() {
        let f1 = vec![1.0, 2.0];
        assert!(composite_scores(&[&f1], &[0.0], 1, 2).is_err());
        assert!(composite_scores(&[&f1], &[-1.0], 1, 2).is_err());
        assert!(composite_scores(&[&f1], &[f64::NAN], 1, 2).is_err());
        assert!(composite_scores(&[&f1], &[0.5, 0.5], 1, 2).is_err());
    }

    #[test]
    fn momentum_refuses_bad_windows() {
        let p = vec![1.0; 10];
        assert!(momentum_panel(&p, 10, 1, 0, 0).is_err());
        assert!(momentum_panel(&p, 10, 1, 5, 5).is_err());
    }
}
