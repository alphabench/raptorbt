//! Python bindings for the expression language: compile once, evaluate over whole columns or
//! stream one bar at a time. One evaluator serves every function in the registry, so a new
//! function needs no binding.
//!
//! The Python surface is `raptorbt.expressions`; these classes are its engine.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::Mutex;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::errors::Fail;
use super::numpy_bridge::numpy_cow_f64;
use crate::quant::eval::{eval, Bar, EvalError, Inputs, Plan, Stream};
use crate::quant::lang::{compile, Env, ExprError, Field, Op, Program};
use crate::quant::registry::expr_registry_json as registry_json;

/// A compile refusal as the structured error: the position is a character offset.
fn compile_fail(source: &str, e: ExprError) -> PyErr {
    let position = source[..e.pos].chars().count();
    Fail::new(e.code.as_str(), e.message)
        .op("expr")
        .param("expression")
        .value(source)
        .valid(e.valid)
        .at(position, e.snippet)
        .into()
}

fn eval_fail(operation: &str, e: EvalError) -> PyErr {
    let text = e.to_string();
    let fail = match e {
        EvalError::Unsupported(_) => Fail::new("expr_unsupported_here", text).param("expression"),
        EvalError::NoInput => Fail::new("missing_field", text).param("data"),
        EvalError::MissingField(name) => Fail::new("missing_field", text).param(name),
        EvalError::MissingHandle(name) => Fail::new("missing_field", text).param(&name),
        EvalError::LengthMismatch { ref name, .. } => {
            Fail::new("length_mismatch", text).param(name)
        }
        EvalError::HandleCount { .. } | EvalError::Node(_) => Fail::new("invalid_input", text),
    };
    fail.op(operation).into()
}

/// A checked expression and the plan that runs it. Immutable: evaluation and streams take
/// their own state.
#[pyclass(name = "ExprProgram", frozen)]
pub struct PyExprProgram {
    program: Program,
    plan: Plan,
    source: String,
}

impl PyExprProgram {
    /// This program's id in the bank's interner: equal canonical text, equal id.
    pub(crate) fn intern(&self) -> crate::core::Result<crate::runtime::exprs::ProgramId> {
        crate::runtime::exprs::intern(&self.program, &self.plan)
    }
}

#[pymethods]
impl PyExprProgram {
    /// Compile `source`. `params` binds the `$name` constants the source uses.
    #[new]
    #[pyo3(signature = (source, params=None))]
    fn new(source: &str, params: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut env = Env::default();
        if let Some(params) = params {
            let bound: BTreeMap<String, f64> = params.extract().map_err(|e| {
                PyErr::from(
                    Fail::new(
                        "invalid_parameter",
                        format!("params must map names to numbers: {e}"),
                    )
                    .op("expr")
                    .param("params"),
                )
            })?;
            env.params = bound;
        }
        let program = compile(source, &env).map_err(|e| compile_fail(source, e))?;
        let plan = Plan::new(&program).map_err(|e| eval_fail("expr", e))?;
        Ok(Self { program, plan, source: source.to_string() })
    }

    #[getter]
    fn source(&self) -> &str {
        &self.source
    }

    /// The canonical text: equal for every spelling of one expression.
    #[getter]
    fn canonical(&self) -> &str {
        &self.program.canonical
    }

    /// Lowercase hex sha256 of the canonical text and the revisions of the functions it calls.
    #[getter]
    fn hash(&self) -> &str {
        &self.program.hash
    }

    /// Leading bars that are unknown (NaN) on clean data.
    #[getter]
    fn warmup(&self) -> usize {
        self.program.warmup()
    }

    /// Bars until infinite-memory functions have forgotten their seed to within `eps`.
    fn settle(&self, eps: f64) -> PyResult<usize> {
        if !(eps > 0.0 && eps < 1.0) {
            return Err(Fail::new("invalid_parameter", "eps must be a number in (0, 1)")
                .op("ExprProgram.settle")
                .param("eps")
                .value(eps)
                .valid(["a number in (0, 1), e.g. 1e-6"])
                .into());
        }
        Ok(self.program.settle(eps))
    }

    /// The names of the output columns, in the order `evaluate` returns them.
    #[getter]
    fn outputs(&self) -> Vec<String> {
        let dag = &self.program.dag;
        let name_of = |op: &Op, index: usize| match op {
            Op::Call { def, .. } => def.outputs[index].to_string(),
            _ => "value".to_string(),
        };
        self.program
            .outputs()
            .into_iter()
            .map(|(id, i)| match &dag.node(id).op {
                Op::Out { call, index } => name_of(&dag.node(*call).op, *index as usize),
                op => name_of(op, i as usize),
            })
            .collect()
    }

    /// The columns the program reads, sorted.
    #[getter]
    fn fields(&self) -> Vec<&'static str> {
        let mut names: Vec<&'static str> = self.plan.fields().iter().map(|f| f.name()).collect();
        names.sort_unstable();
        names.dedup();
        names
    }

    /// Evaluate over whole columns. `data` is a dict of float64 numpy arrays of one length keyed
    /// by field name; keys that are not fields are ignored. Returns one array per output.
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Vec<Bound<'py, PyArray1<f64>>>> {
        const OP: &str = "ExprProgram.evaluate";
        let dict = data.cast::<PyDict>().map_err(|_| {
            PyErr::from(
                Fail::new(
                    "invalid_input_type",
                    format!(
                        "data must be a dict of float64 numpy arrays, got {}",
                        data.get_type().name().map(|n| n.to_string()).unwrap_or_default()
                    ),
                )
                .op(OP)
                .param("data"),
            )
        })?;
        let mut held: Vec<(Field, PyReadonlyArray1<'py, f64>)> = Vec::new();
        for f in Field::ALL {
            let Some(v) = dict.get_item(f.name())? else { continue };
            let arr = v.extract::<PyReadonlyArray1<'py, f64>>().map_err(|_| {
                PyErr::from(
                    Fail::new(
                        "invalid_array",
                        format!(
                            "'{}' must be a 1-D numpy float64 array, got {}; convert with \
                             np.asarray(x, dtype=np.float64)",
                            f.name(),
                            v.get_type().name().map(|n| n.to_string()).unwrap_or_default()
                        ),
                    )
                    .op(OP)
                    .param(f.name()),
                )
            })?;
            held.push((f, arr));
        }
        let cols: Vec<(Field, Cow<'_, [f64]>)> =
            held.iter().map(|(f, a)| (*f, numpy_cow_f64(a))).collect();
        let result = py.detach(|| {
            let mut inputs = Inputs::new();
            for (f, c) in &cols {
                inputs = inputs.field(*f, c);
            }
            eval(&self.plan, &inputs)
        });
        let columns = result.map_err(|e| eval_fail(OP, e))?;
        Ok(columns.into_iter().map(|v| PyArray1::from_vec(py, v)).collect())
    }

    /// A streaming evaluator over this program: one bar in, the outputs for that bar out.
    fn stream(&self) -> PyResult<PyExprStream> {
        let inner = Stream::new(&self.plan).map_err(|e| eval_fail("ExprProgram.stream", e))?;
        Ok(PyExprStream { inner: Mutex::new(inner) })
    }
}

/// Bar-by-bar evaluation of a program. Bit-identical to `ExprProgram.evaluate` on the same bars.
#[pyclass(name = "ExprStream", frozen)]
pub struct PyExprStream {
    inner: Mutex<Stream>,
}

impl PyExprStream {
    fn locked(&self) -> PyResult<std::sync::MutexGuard<'_, Stream>> {
        self.inner.lock().map_err(|_| PyRuntimeError::new_err("expression stream lock poisoned"))
    }
}

#[pymethods]
impl PyExprStream {
    /// Feed one bar; returns one value per output (NaN while unknown).
    fn update(
        &self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> PyResult<Vec<f64>> {
        let mut s = self.locked()?;
        let out = s
            .update(&Bar { open, high, low, close, volume }, &[])
            .map_err(|e| eval_fail("ExprStream.update", e))?;
        Ok(out.to_vec())
    }

    /// True once the outputs can be known: more bars than the warm-up.
    #[getter]
    fn ready(&self) -> PyResult<bool> {
        Ok(self.locked()?.ready())
    }

    /// Bars consumed so far.
    #[getter]
    fn bars(&self) -> PyResult<usize> {
        Ok(self.locked()?.bars())
    }

    /// Forget every bar.
    fn reset(&self) -> PyResult<()> {
        self.locked()?.reset().map_err(|e| eval_fail("ExprStream.reset", e))
    }
}

/// The function registry as JSON text; `raptorbt.expressions` parses it.
#[pyfunction]
pub fn expr_registry_json() -> String {
    registry_json().to_string()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyExprProgram>()?;
    m.add_class::<PyExprStream>()?;
    m.add_function(wrap_pyfunction!(expr_registry_json, m)?)?;
    Ok(())
}
