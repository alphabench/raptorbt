//! Numerics the engine owns rather than borrows from a platform library, so two
//! machines give the same bits. `fsum` is Python's `math.fsum`: a sum taken with it in
//! Python lands on the same float here. The normal distribution is `pricing::norm`.
//!
//! The statistics build on three files, each with its own typed error: `linalg` (QR, Cholesky,
//! Jacobi eigendecomposition and SVD, solve, least squares), `special` (`ln_gamma`, incomplete
//! beta and gamma, the t, chi-squared and F distributions) and `optimize` (BFGS).

pub(crate) mod fsum;
pub mod linalg;
pub mod optimize;
pub mod special;
