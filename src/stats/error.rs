//! The error every statistic in this module tree refuses with. Three variants carry an error
//! code of their own (`degenerate_series`, `insufficient_observations`,
//! `did_not_converge`); a non-finite value and a malformed argument are `invalid_input`.

use thiserror::Error;

use crate::math::linalg::LinAlgError;
use crate::math::special::SpecialError;

/// Why a statistic refused its input.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum StatsError {
    /// Too few observations. Error code `insufficient_observations`.
    #[error("{what}: {found} observations, this statistic needs {needed}")]
    InsufficientObservations { what: String, found: usize, needed: usize },
    /// The data cannot support the statistic: constant, collinear or singular. Error code
    /// `degenerate_series`.
    #[error("{0}")]
    DegenerateSeries(String),
    /// A fit did not settle. Error code `did_not_converge`.
    #[error("{0}")]
    DidNotConverge(String),
    /// A NaN or an infinite value in the input. Error code `invalid_input`.
    #[error("{what} holds a NaN or infinite value")]
    NonFinite { what: String },
    /// An argument outside what the statistic takes (series of unequal length, too many series).
    /// Error code `invalid_input`.
    #[error("{0}")]
    Invalid(String),
}

impl StatsError {
    /// The error code (see `python/raptorbt/errors.py`).
    pub fn code(&self) -> &'static str {
        match self {
            StatsError::InsufficientObservations { .. } => "insufficient_observations",
            StatsError::DegenerateSeries(_) => "degenerate_series",
            StatsError::DidNotConverge(_) => "did_not_converge",
            StatsError::NonFinite { .. } | StatsError::Invalid(_) => "invalid_input",
        }
    }
}

impl From<LinAlgError> for StatsError {
    fn from(e: LinAlgError) -> Self {
        match e {
            LinAlgError::NoConvergence { .. } => StatsError::DidNotConverge(e.to_string()),
            LinAlgError::Singular { .. } | LinAlgError::NotPositiveDefinite { .. } => {
                StatsError::DegenerateSeries(e.to_string())
            }
            other => StatsError::Invalid(other.to_string()),
        }
    }
}

impl From<SpecialError> for StatsError {
    fn from(e: SpecialError) -> Self {
        StatsError::Invalid(e.to_string())
    }
}
