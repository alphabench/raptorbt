//! Error types for RaptorBT.

use thiserror::Error;

/// Result type alias for RaptorBT operations.
pub type Result<T> = std::result::Result<T, RaptorError>;

/// Error types for the backtesting engine.
#[derive(Error, Debug)]
pub enum RaptorError {
    /// Data length mismatch between arrays.
    #[error("Data length mismatch: expected {expected}, got {actual}")]
    LengthMismatch { expected: usize, actual: usize },

    /// Invalid parameter value.
    #[error("Invalid parameter: {message}")]
    InvalidParameter { message: String },

    /// Insufficient data for calculation.
    #[error("Insufficient data: need at least {required} elements, got {available}")]
    InsufficientData { required: usize, available: usize },

    /// Invalid configuration.
    #[error("Invalid configuration: {message}")]
    InvalidConfig { message: String },

    /// Division by zero error.
    #[error("Division by zero in {context}")]
    DivisionByZero { context: String },

    /// Empty data error.
    #[error("Empty data provided for {context}")]
    EmptyData { context: String },

    /// Invalid index access.
    #[error("Index {index} out of bounds for length {length}")]
    IndexOutOfBounds { index: usize, length: usize },

    /// Python conversion error.
    #[error("Python conversion error: {message}")]
    PythonError { message: String },
}

impl RaptorError {
    /// Create a length mismatch error.
    pub fn length_mismatch(expected: usize, actual: usize) -> Self {
        Self::LengthMismatch { expected, actual }
    }

    /// Create an invalid parameter error.
    pub fn invalid_parameter(message: impl Into<String>) -> Self {
        Self::InvalidParameter {
            message: message.into(),
        }
    }

    /// Create an insufficient data error.
    pub fn insufficient_data(required: usize, available: usize) -> Self {
        Self::InsufficientData {
            required,
            available,
        }
    }

    /// Create an invalid config error.
    pub fn invalid_config(message: impl Into<String>) -> Self {
        Self::InvalidConfig {
            message: message.into(),
        }
    }

    /// Create a division by zero error.
    pub fn division_by_zero(context: impl Into<String>) -> Self {
        Self::DivisionByZero {
            context: context.into(),
        }
    }

    /// Create an empty data error.
    pub fn empty_data(context: impl Into<String>) -> Self {
        Self::EmptyData {
            context: context.into(),
        }
    }
}

impl From<RaptorError> for pyo3::PyErr {
    fn from(err: RaptorError) -> pyo3::PyErr {
        pyo3::exceptions::PyValueError::new_err(err.to_string())
    }
}
