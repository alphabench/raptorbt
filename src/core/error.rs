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

    /// A price column holds a NaN or an infinity.
    #[error("{label}: {column} is not finite ({value}) at index {index}")]
    NonFinitePrice { label: String, column: &'static str, index: usize, value: f64 },

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
        Self::InvalidParameter { message: message.into() }
    }

    /// Create an invalid config error.
    pub fn invalid_config(message: impl Into<String>) -> Self {
        Self::InvalidConfig { message: message.into() }
    }

    /// Create a division by zero error.
    pub fn division_by_zero(context: impl Into<String>) -> Self {
        Self::DivisionByZero { context: context.into() }
    }

    /// Create an empty data error.
    pub fn empty_data(context: impl Into<String>) -> Self {
        Self::EmptyData { context: context.into() }
    }
}

impl From<RaptorError> for pyo3::PyErr {
    fn from(err: RaptorError) -> pyo3::PyErr {
        use crate::python::errors::Fail;
        let code = match &err {
            RaptorError::LengthMismatch { .. } => "length_mismatch",
            RaptorError::InvalidParameter { .. } => "invalid_parameter",
            RaptorError::InvalidConfig { .. } => "invalid_config",
            RaptorError::EmptyData { .. } => "empty_data",
            RaptorError::NonFinitePrice { .. } => "non_finite_price",
            RaptorError::DivisionByZero { .. }
            | RaptorError::IndexOutOfBounds { .. }
            | RaptorError::PythonError { .. } => "invalid_data",
        };
        let mut text = err.to_string();
        let mut code = code;
        let mut valid: Vec<String> = Vec::new();
        // A refusal from deep in the runtime that the structured error knows by name: the message
        // starts `expr_type: ` and ends `valid: a, b`.
        if let Some(rest) = text.strip_prefix("Invalid parameter: expr_type: ") {
            if let Some((_, list)) = rest.rsplit_once("valid: ") {
                valid = list.split(", ").map(str::to_string).collect();
            }
            code = "expr_type";
            text = rest.to_string();
        }
        let mut fail = Fail::new(code, text);
        if !valid.is_empty() {
            fail = fail.valid(valid);
        }
        match err {
            RaptorError::LengthMismatch { actual, .. } => fail.value(actual),
            RaptorError::NonFinitePrice { column, index, .. } => fail.param(column).value(index),
            _ => fail,
        }
        .into()
    }
}
