//! Errors for the portfolio math surface.
//!
//! Every function in `covariance`, `optimize`, `factor_panel`, `risk_contrib`
//! and `rebalance` refuses malformed or degenerate input with one of these
//! variants instead of clamping, defaulting, or silently repairing it. This is
//! a financial library: a wrong number that looks plausible is strictly worse
//! than an error, so the bias is always toward refusal.

use thiserror::Error;

/// Error type shared by the portfolio math modules.
#[derive(Debug, Error)]
pub enum PortfolioMathError {
    /// A NaN or infinity where a finite value is required.
    #[error("non-finite value at row {row}, col {col}")]
    NonFinite { row: usize, col: usize },

    /// Array lengths or matrix dimensions do not agree.
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),

    /// Asset identifier lists do not agree (content or order).
    #[error("asset ids mismatch: {0}")]
    AssetIdMismatch(String),

    /// An asset has zero return variance, which breaks the correlation
    /// target of the shrinkage estimator.
    #[error("zero-variance asset at index {0}")]
    ZeroVariance(usize),

    /// A matrix that must be positive definite is not.
    #[error("covariance not positive definite: {0}")]
    NotPositiveDefinite(String),

    /// The constraint set admits no solution.
    #[error("infeasible constraints: {0}")]
    Infeasible(String),

    /// The solver terminated without a certified solution.
    #[error("solver failed: {0}")]
    SolverFailed(String),

    /// A cross-section has fewer usable names than the caller's minimum.
    #[error("cross-section too small: {have} < min_names {need}")]
    TooFewNames { have: usize, need: usize },

    /// Input that is structurally valid but numerically degenerate
    /// (e.g. zero portfolio volatility, empty panel).
    #[error("degenerate input: {0}")]
    DegenerateInput(String),
}

/// Validate that every value in a row-major panel is finite.
pub(crate) fn require_finite(
    values: &[f64],
    n_rows: usize,
    n_cols: usize,
) -> Result<(), PortfolioMathError> {
    if values.len() != n_rows * n_cols {
        return Err(PortfolioMathError::ShapeMismatch(format!(
            "expected {n_rows}x{n_cols}={} values, got {}",
            n_rows * n_cols,
            values.len()
        )));
    }
    for (idx, v) in values.iter().enumerate() {
        if !v.is_finite() {
            return Err(PortfolioMathError::NonFinite {
                row: idx / n_cols,
                col: idx % n_cols,
            });
        }
    }
    Ok(())
}
