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

fn check_shape(
    values: &[f64],
    n_dates: usize,
    n_assets: usize,
) -> Result<(), PortfolioMathError> {
    if values.len() != n_dates * n_assets {
        return Err(PortfolioMathError::ShapeMismatch(format!(
            "expected {n_dates}x{n_assets}={} values, got {}",
            n_dates * n_assets,
            values.len()
        )));
    }
    for (idx, v) in values.iter().enumerate() {
        if v.is_infinite() {
            return Err(PortfolioMathError::NonFinite {
                row: idx / n_assets,
                col: idx % n_assets,
            });
        }
    }
    Ok(())
}

/// Indices of finite values in one date row.
fn finite_cols(row: &[f64]) -> Vec<usize> {
    row.iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .map(|(i, _)| i)
        .collect()
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
        return Err(PortfolioMathError::DegenerateInput(
            "no factor panels supplied".into(),
        ));
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

#[cfg(test)]
mod tests {
    use super::*;

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
