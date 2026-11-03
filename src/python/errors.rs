//! Structured errors raised from the bindings.
//!
//! A [`Fail`] names a code from `raptorbt.errors.CODES`; the Python side picks
//! the exception class (always a subclass of the builtin that used to be
//! raised) and fills in the alternatives and documentation domain registered
//! for that code. The first line of the message is unchanged.

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyDict;

static MAKE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// A supplied value, handed to Python as the matching builtin.
#[derive(Debug, Clone)]
pub enum Val {
    Str(String),
    Int(i64),
    Float(f64),
    List(Vec<String>),
}

impl From<&str> for Val {
    fn from(v: &str) -> Self {
        Val::Str(v.to_string())
    }
}
impl From<String> for Val {
    fn from(v: String) -> Self {
        Val::Str(v)
    }
}
impl From<i64> for Val {
    fn from(v: i64) -> Self {
        Val::Int(v)
    }
}
impl From<i32> for Val {
    fn from(v: i32) -> Self {
        Val::Int(v.into())
    }
}
impl From<usize> for Val {
    fn from(v: usize) -> Self {
        Val::Int(v as i64)
    }
}
impl From<f64> for Val {
    fn from(v: f64) -> Self {
        Val::Float(v)
    }
}

/// A structured error under construction.
#[derive(Debug, Clone)]
pub struct Fail {
    code: &'static str,
    message: String,
    operation: Option<String>,
    parameter: Option<String>,
    value: Option<Val>,
    valid: Vec<String>,
    skill: Option<&'static str>,
}

impl Fail {
    pub fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            operation: None,
            parameter: None,
            value: None,
            valid: Vec::new(),
            skill: None,
        }
    }

    /// The call or step that failed.
    pub fn op(mut self, operation: &str) -> Self {
        self.operation = Some(operation.to_string());
        self
    }

    /// The argument that was invalid.
    pub fn param(mut self, parameter: &str) -> Self {
        self.parameter = Some(parameter.to_string());
        self
    }

    /// What was supplied.
    pub fn value(mut self, value: impl Into<Val>) -> Self {
        self.value = Some(value.into());
        self
    }

    /// Alternatives, when they depend on the call (otherwise the registry's apply).
    pub fn valid<S: ToString>(mut self, valid: impl IntoIterator<Item = S>) -> Self {
        self.valid = valid.into_iter().map(|s| s.to_string()).collect();
        self
    }

    /// The skill that documents this refusal, when it is not the code's default
    /// (the registry gives `invalid_parameter` the `core` skill).
    pub fn skill(mut self, skill: &'static str) -> Self {
        self.skill = Some(skill);
        self
    }

    fn build(&self, py: Python<'_>) -> PyResult<PyErr> {
        let make = MAKE.get_or_try_init(py, || {
            Ok::<_, PyErr>(py.import("raptorbt.errors")?.getattr("make")?.unbind())
        })?;
        let kw = PyDict::new(py);
        if let Some(v) = &self.operation {
            kw.set_item("operation", v)?;
        }
        if let Some(v) = &self.parameter {
            kw.set_item("parameter", v)?;
        }
        if let Some(v) = &self.value {
            match v {
                Val::Str(x) => kw.set_item("value", x)?,
                Val::Int(x) => kw.set_item("value", x)?,
                Val::Float(x) => kw.set_item("value", x)?,
                Val::List(x) => kw.set_item("value", x)?,
            }
        }
        if !self.valid.is_empty() {
            kw.set_item("valid", &self.valid)?;
        }
        if let Some(skill) = self.skill {
            kw.set_item("skill", skill)?;
        }
        let exc = make.bind(py).call((self.code, &self.message), Some(&kw))?;
        Ok(PyErr::from_value(exc))
    }
}

impl From<Fail> for PyErr {
    fn from(f: Fail) -> PyErr {
        Python::attach(|py| f.build(py).unwrap_or_else(|e| e))
    }
}

/// Run `f`, turning a money fault (an amount that cannot be booked) into the
/// structured `money_overflow` error. The kernel signals it by panicking; a
/// constructor cannot rely on the Python-side guard, which wraps methods and
/// functions but not `__new__`. Any other panic keeps unwinding.
pub fn guard_money<T>(operation: &str, f: impl FnOnce() -> T) -> PyResult<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(|payload| {
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied());
        match message.and_then(|m| m.strip_prefix(crate::accounts::money::MONEY_FAULT_PREFIX)) {
            Some(detail) => PyErr::from(
                Fail::new(
                    "money_overflow",
                    format!("a booked amount cannot be represented: {}", detail.trim()),
                )
                .op(operation),
            ),
            None => std::panic::resume_unwind(payload),
        }
    })
}

/// An argument or array that a computation refuses; the message is the
/// computation's own.
pub fn invalid_input(message: impl std::fmt::Display) -> PyErr {
    Fail::new("invalid_input", message.to_string()).into()
}

/// Name the call on a refusal that names none (a constructor, whose frame the Python guard does
/// not wrap). A refusal that already names its operation keeps it.
pub fn with_op(err: PyErr, operation: &str) -> PyErr {
    Python::attach(|py| {
        let value = err.value(py);
        if value.getattr("operation").is_ok_and(|op| op.is_none()) {
            let _ = value.setattr("operation", operation);
        }
    });
    err
}
