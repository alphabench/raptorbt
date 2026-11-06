//! The error every statistic in this module tree refuses with. Three variants carry a platform
//! error code (`degenerate_series`, `insufficient_observations`, `did_not_converge`); the others
//! are argument errors the tool layer words itself.

use thiserror::Error;

use crate::math::linalg::LinAlgError;

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
    /// A NaN or an infinite value in the input.
    #[error("{what} holds a NaN or infinite value")]
    NonFinite { what: String },
    /// An argument outside what the statistic takes (series of unequal length, too many series).
    #[error("{0}")]
    Invalid(String),
}

impl StatsError {
    /// The error code, where the statistic has one.
    pub fn code(&self) -> Option<&'static str> {
        match self {
            StatsError::InsufficientObservations { .. } => Some("insufficient_observations"),
            StatsError::DegenerateSeries(_) => Some("degenerate_series"),
            StatsError::DidNotConverge(_) => Some("did_not_converge"),
            StatsError::NonFinite { .. } | StatsError::Invalid(_) => None,
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
