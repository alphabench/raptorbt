//! Python bindings for the expression language: compile once, evaluate over whole columns or
//! stream one bar at a time. One evaluator serves every function in the registry, so a new
//! function needs no binding.
//!
//! The Python surface is `raptorbt.expressions`; these classes are its engine.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::Mutex;

use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::errors::Fail;
use super::numpy_bridge::numpy_cow_f64;
use super::schedule_bindings::PySessionSchedule;
use crate::data::BarStamp;
use crate::quant::eval::{
    eval, eval_panel, Bar, ClockCtx, EvalError, Inputs, PanelInputs, PanelPlan, Plan, Stream,
};
use crate::quant::factor::{factor_report, FactorParams};
use crate::quant::lang::{compile, Env, ExprError, Field, Handle, Op, Program, Ty};
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
        EvalError::NoTimestamps => Fail::new("missing_field", text).param("timestamps"),
        EvalError::NoClock => Fail::new("missing_bar_stamp", text).param("bar_stamp"),
        EvalError::Timeframe { valid, .. } => {
            Fail::new("expr_type", text).param("expression").valid(valid)
        }
    };
    fail.op(operation).into()
}

/// The clock an `at(...)` is cut by, from the arguments of `evaluate`, `stream` and `bound_hash`.
/// None when no `bar_stamp` was given: a program with `at(...)` then refuses to run.
fn clock_from(
    operation: &str,
    bar_stamp: Option<&str>,
    tz_offset_ns: i64,
    session: Option<&PySessionSchedule>,
    bar_ns: Option<i64>,
) -> PyResult<Option<ClockCtx>> {
    let Some(text) = bar_stamp else { return Ok(None) };
    let stamp = BarStamp::parse(text).map_err(|e| {
        PyErr::from(
            Fail::new("invalid_bar_stamp", e.to_string())
                .op(operation)
                .param("bar_stamp")
                .value(text)
                .valid(["open", "close"]),
        )
    })?;
    let mut clock = ClockCtx::new(tz_offset_ns, stamp, session.map(|s| s.inner.clone()));
    clock.base_ns = bar_ns;
    Ok(Some(clock))
}

/// A float64 panel, borrowed when it is C-contiguous and copied in logical order when not.
fn panel_cow<'a>(a: &'a PyReadonlyArray2<'_, f64>) -> Cow<'a, [f64]> {
    match a.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(a.as_array().iter().copied().collect()),
    }
}

fn shape_fail(operation: &str, name: &str, got: &[usize], want: (usize, usize)) -> PyErr {
    Fail::new(
        "length_mismatch",
        format!(
            "'{name}' has shape {got:?}, expected ({}, {}) as the other panels",
            want.0, want.1
        ),
    )
    .op(operation)
    .param(name)
    .into()
}

fn not_a_panel(operation: &str, name: &str, got: &Bound<'_, PyAny>, what: &str) -> PyErr {
    Fail::new(
        "invalid_array",
        format!(
            "'{name}' must be {what}, got {}; convert with np.asarray(x, dtype=np.float64)",
            got.get_type().name().map(|n| n.to_string()).unwrap_or_default()
        ),
    )
    .op(operation)
    .param(name)
    .into()
}

/// A group series: a full panel, or one value per name repeated on every date.
enum Labels<'py> {
    Full(PyReadonlyArray2<'py, f64>),
    PerName(PyReadonlyArray1<'py, f64>),
}

/// A checked expression and the plan that runs it. Immutable: evaluation and streams take
/// their own state.
#[pyclass(name = "ExprProgram", frozen)]
pub struct PyExprProgram {
    program: Program,
    /// The series plan; none for a panel program (a cross-sectional call), which has only
    /// `panel`.
    plan: Option<Plan>,
    /// The panel plan; an error for a program with no field to make a panel of (a constant).
    panel: Result<PanelPlan, EvalError>,
    source: String,
}

impl PyExprProgram {
    /// This program's id in the bank's interner: equal canonical text, equal id. A panel
    /// program has none: sealing one on the bank is a later slice.
    pub(crate) fn intern(&self) -> crate::core::Result<crate::runtime::exprs::ProgramId> {
        match &self.plan {
            Some(plan) => crate::runtime::exprs::intern(&self.program, plan),
            None => Err(crate::core::RaptorError::invalid_parameter(
                "a cross-sectional expression has a value only over a panel (Expr.eval_panel); \
                 sealing one on the indicator bank is not supported yet",
            )),
        }
    }

    /// Whether this is a cross-sectional program (no series plan).
    pub(crate) fn is_cross_sectional(&self) -> bool {
        self.plan.is_none()
    }

    /// The series plan, or the refusal for a panel program.
    fn series(&self, operation: &str) -> PyResult<&Plan> {
        self.plan.as_ref().ok_or_else(|| {
            Fail::new(
                "expr_unsupported_here",
                "this expression is cross-sectional: it has a value only over a panel, so \
                 evaluate it with eval_panel (it cannot be evaluated or streamed one series at \
                 a time)",
            )
            .op(operation)
            .param("expression")
            .into()
        })
    }
}

#[pymethods]
impl PyExprProgram {
    /// Compile `source`. `params` binds the `$name` constants the source uses.
    ///
    /// `groups` names series the caller supplies per name and date to `eval_panel` (a sector
    /// code for `group_demean`); they read as numbers.
    #[new]
    #[pyo3(signature = (source, params=None, groups=None))]
    fn new(
        source: &str,
        params: Option<&Bound<'_, PyDict>>,
        groups: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let mut env = Env::default();
        for g in groups.unwrap_or_default() {
            env.series.insert(g, Handle { ty: Ty::series(), warm: 0 });
        }
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
        let panel = PanelPlan::new(&program);
        let plan = if program.is_panel() {
            if let Err(e) = &panel {
                return Err(eval_fail("expr", e.clone()));
            }
            None
        } else {
            Some(Plan::new(&program).map_err(|e| eval_fail("expr", e))?)
        };
        Ok(Self { program, plan, panel, source: source.to_string() })
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

    /// Leading bars that are unknown (NaN) on clean data. None when the program uses `at(...)`:
    /// the base bars a coarser clock costs depend on how many bars a window holds. See
    /// `warmup_by_clock` and `warmup_bars`.
    #[getter]
    fn warmup(&self) -> Option<usize> {
        (!self.program.has_at()).then(|| self.program.warmup())
    }

    /// Warm-up per clock: the base clock's, then each `at(...)` timeframe's in bars of that
    /// timeframe.
    #[getter]
    fn warmup_by_clock(&self) -> Vec<(String, usize)> {
        self.program.clock_warmups()
    }

    /// The most leading unknown base bars, given the base bar width in nanoseconds; None when
    /// that cannot be known (no width, or a month or a year window).
    #[pyo3(signature = (bar_ns=None))]
    fn warmup_bars(&self, bar_ns: Option<i64>) -> Option<usize> {
        self.program.warmup_bars(bar_ns)
    }

    /// Whether the program uses `at(...)`, so evaluating it needs timestamps and a bar stamp.
    #[getter]
    fn uses_at(&self) -> bool {
        self.program.has_at()
    }

    /// Whether the program has a cross-sectional call: it has a value only over a panel.
    #[getter]
    fn is_panel(&self) -> bool {
        self.program.is_panel()
    }

    /// The program hash with the clock folded in when the program uses `at(...)`: the zone, the
    /// bar stamp and the session decide its numbers, so they are part of its identity. Equal to
    /// `hash` for a program without `at`.
    #[pyo3(signature = (bar_stamp=None, tz_offset_ns=0, session=None))]
    fn bound_hash(
        &self,
        bar_stamp: Option<&str>,
        tz_offset_ns: i64,
        session: Option<&PySessionSchedule>,
    ) -> PyResult<String> {
        if !self.program.has_at() {
            return Ok(self.program.hash.clone());
        }
        let Some(clock) =
            clock_from("ExprProgram.bound_hash", bar_stamp, tz_offset_ns, session, None)?
        else {
            return Err(eval_fail("ExprProgram.bound_hash", EvalError::NoClock));
        };
        Ok(self.program.bound_hash(&clock.text()))
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
        let fields = match (&self.plan, &self.panel) {
            (Some(plan), _) => plan.fields(),
            (None, Ok(panel)) => panel.fields().to_vec(),
            (None, Err(_)) => Vec::new(),
        };
        let mut names: Vec<&'static str> = fields.iter().map(|f| f.name()).collect();
        names.sort_unstable();
        names.dedup();
        names
    }

    /// Evaluate over whole columns. `data` is a dict of float64 numpy arrays of one length keyed
    /// by field name; keys that are not fields are ignored. Returns one array per output.
    ///
    /// A program that uses `at(...)` also reads `data["timestamps"]` (an int64 array of
    /// nanoseconds) and needs `bar_stamp` ("open" or "close": what a timestamp means); the zone
    /// daily windows are cut in is `tz_offset_ns` (an instrument's `session` replaces it), and
    /// `bar_ns`, the base bar width, lets a timeframe that is not coarser be refused. A program
    /// without `at` ignores all four.
    #[pyo3(signature = (data, bar_stamp=None, tz_offset_ns=0, session=None, bar_ns=None))]
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        bar_stamp: Option<&str>,
        tz_offset_ns: i64,
        session: Option<&PySessionSchedule>,
        bar_ns: Option<i64>,
    ) -> PyResult<Vec<Bound<'py, PyArray1<f64>>>> {
        const OP: &str = "ExprProgram.evaluate";
        let plan = self.series(OP)?;
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
        let stamps: Option<PyReadonlyArray1<'py, i64>> = if plan.has_at() {
            match dict.get_item("timestamps")? {
                None => None,
                Some(v) => Some(v.extract::<PyReadonlyArray1<'py, i64>>().map_err(|_| {
                    PyErr::from(
                        Fail::new(
                            "invalid_array",
                            "'timestamps' must be a 1-D numpy int64 array of nanoseconds",
                        )
                        .op(OP)
                        .param("timestamps"),
                    )
                })?),
            }
        } else {
            None
        };
        let stamps: Option<Cow<'_, [i64]>> = match &stamps {
            Some(a) => Some(match a.as_slice() {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(a.as_array().iter().copied().collect()),
            }),
            None => None,
        };
        let clock = if plan.has_at() {
            clock_from(OP, bar_stamp, tz_offset_ns, session, bar_ns)?
        } else {
            None
        };
        let result = py.detach(|| {
            let mut inputs = Inputs::new();
            for (f, c) in &cols {
                inputs = inputs.field(*f, c);
            }
            if let Some(ts) = &stamps {
                inputs = inputs.timestamps(ts);
            }
            if let Some(clock) = clock {
                inputs = inputs.clock(clock);
            }
            eval(plan, &inputs)
        });
        let columns = result.map_err(|e| eval_fail(OP, e))?;
        Ok(columns.into_iter().map(|v| PyArray1::from_vec(py, v)).collect())
    }

    /// Evaluate over a panel: `data` maps field names to 2-D float64 arrays of one shape
    /// `(n_dates, n_assets)`; keys that are not fields (or group series) are ignored. Returns one
    /// 2-D array per output.
    ///
    /// `present` is a boolean array of that shape saying which names have a bar at each date; an
    /// absent bar does not step a time-series function, so a name listed late counts its
    /// warm-up from its own first bar. Without it a cell is absent where every field the
    /// expression reads is NaN. A group series (see `groups` in the constructor) is a 2-D
    /// array, or a 1-D array with one value per name.
    ///
    /// An expression with `at(...)` also reads `data["timestamps"]` (int64 nanoseconds, one per
    /// date, shared by the names) and needs `bar_stamp` and the optional zone arguments as for
    /// `evaluate`. The GIL is released while it runs.
    #[pyo3(signature = (data, present=None, bar_stamp=None, tz_offset_ns=0, session=None, bar_ns=None))]
    #[allow(clippy::too_many_arguments)]
    fn eval_panel<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        present: Option<&Bound<'py, PyAny>>,
        bar_stamp: Option<&str>,
        tz_offset_ns: i64,
        session: Option<&PySessionSchedule>,
        bar_ns: Option<i64>,
    ) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
        const OP: &str = "ExprProgram.eval_panel";
        let plan = self.panel.as_ref().map_err(|e| eval_fail(OP, e.clone()))?;
        let dict = data.cast::<PyDict>().map_err(|_| {
            PyErr::from(
                Fail::new(
                    "invalid_input_type",
                    format!(
                        "data must be a dict of 2-D float64 numpy arrays, got {}",
                        data.get_type().name().map(|n| n.to_string()).unwrap_or_default()
                    ),
                )
                .op(OP)
                .param("data"),
            )
        })?;
        const WANT2: &str = "a 2-D numpy float64 array (n_dates, n_assets)";
        let mut held: Vec<(Field, PyReadonlyArray2<'py, f64>)> = Vec::new();
        for f in Field::ALL {
            let Some(v) = dict.get_item(f.name())? else { continue };
            let arr = v
                .extract::<PyReadonlyArray2<'py, f64>>()
                .map_err(|_| not_a_panel(OP, f.name(), &v, WANT2))?;
            held.push((f, arr));
        }
        let mut labels: Vec<(String, Labels<'py>)> = Vec::new();
        for name in plan.handles() {
            let Some(v) = dict.get_item(name)? else { continue };
            let l = if let Ok(a) = v.extract::<PyReadonlyArray2<'py, f64>>() {
                Labels::Full(a)
            } else if let Ok(a) = v.extract::<PyReadonlyArray1<'py, f64>>() {
                Labels::PerName(a)
            } else {
                return Err(not_a_panel(
                    OP,
                    name,
                    &v,
                    "a 2-D (n_dates, n_assets) or 1-D (n_assets) float64 array",
                ));
            };
            labels.push((name.clone(), l));
        }
        let shape_of = |a: &[usize]| (a[0], a[1]);
        let (n, m) = if let Some((_, a)) = held.first() {
            shape_of(a.shape())
        } else if let Some(Labels::Full(a)) = labels.iter().map(|(_, l)| l).next() {
            shape_of(a.shape())
        } else {
            return Err(eval_fail(OP, EvalError::NoInput));
        };
        for (f, a) in &held {
            if a.shape() != [n, m] {
                return Err(shape_fail(OP, f.name(), a.shape(), (n, m)));
            }
        }
        let cols: Vec<(Field, Cow<'_, [f64]>)> =
            held.iter().map(|(f, a)| (*f, panel_cow(a))).collect();
        let mut label_cols: Vec<(&str, Cow<'_, [f64]>)> = Vec::new();
        for (name, l) in &labels {
            let cow = match l {
                Labels::Full(a) => {
                    if a.shape() != [n, m] {
                        return Err(shape_fail(OP, name, a.shape(), (n, m)));
                    }
                    panel_cow(a)
                }
                Labels::PerName(a) => {
                    if a.len() != m {
                        return Err(shape_fail(OP, name, &[a.len()], (n, m)));
                    }
                    let per: Vec<f64> = a.as_array().iter().copied().collect();
                    Cow::Owned((0..n).flat_map(|_| per.iter().copied()).collect())
                }
            };
            label_cols.push((name.as_str(), cow));
        }
        let mask: Option<PyReadonlyArray2<'py, bool>> = match present {
            None => None,
            Some(v) => Some(v.extract::<PyReadonlyArray2<'py, bool>>().map_err(|_| {
                not_a_panel(OP, "present", v, "a 2-D numpy bool array (n_dates, n_assets)")
            })?),
        };
        let mask: Option<Cow<'_, [bool]>> = match &mask {
            None => None,
            Some(a) => {
                if a.shape() != [n, m] {
                    return Err(shape_fail(OP, "present", a.shape(), (n, m)));
                }
                Some(match a.as_slice() {
                    Ok(s) => Cow::Borrowed(s),
                    Err(_) => Cow::Owned(a.as_array().iter().copied().collect()),
                })
            }
        };
        let stamps: Option<PyReadonlyArray1<'py, i64>> = if plan.has_at() {
            match dict.get_item("timestamps")? {
                None => None,
                Some(v) => Some(v.extract::<PyReadonlyArray1<'py, i64>>().map_err(|_| {
                    PyErr::from(
                        Fail::new(
                            "invalid_array",
                            "'timestamps' must be a 1-D numpy int64 array of nanoseconds, one per date",
                        )
                        .op(OP)
                        .param("timestamps"),
                    )
                })?),
            }
        } else {
            None
        };
        let stamps: Option<Cow<'_, [i64]>> = stamps.as_ref().map(|a| match a.as_slice() {
            Ok(s) => Cow::Borrowed(s),
            Err(_) => Cow::Owned(a.as_array().iter().copied().collect()),
        });
        let clock = if plan.has_at() {
            clock_from(OP, bar_stamp, tz_offset_ns, session, bar_ns)?
        } else {
            None
        };
        let result = py.detach(|| {
            let mut inputs = PanelInputs::new(n, m);
            for (f, c) in &cols {
                inputs = inputs.field(*f, c);
            }
            for (name, c) in &label_cols {
                inputs = inputs.handle(name, c);
            }
            if let Some(mk) = &mask {
                inputs = inputs.present(mk);
            }
            if let Some(ts) = &stamps {
                inputs = inputs.timestamps(ts);
            }
            if let Some(clock) = clock {
                inputs = inputs.clock(clock);
            }
            eval_panel(plan, &inputs)
        });
        let panels = result.map_err(|e| eval_fail(OP, e))?;
        panels
            .into_iter()
            .map(|v| {
                Array2::from_shape_vec((n, m), v)
                    .map(|a| a.into_pyarray(py))
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })
            .collect()
    }

    /// A streaming evaluator over this program: one bar in, the outputs for that bar out. A
    /// program that uses `at(...)` needs `bar_stamp` (and optionally `tz_offset_ns`, `session`,
    /// `bar_ns` as for `evaluate`) and is fed with `update_at`.
    #[pyo3(signature = (bar_stamp=None, tz_offset_ns=0, session=None, bar_ns=None))]
    fn stream(
        &self,
        bar_stamp: Option<&str>,
        tz_offset_ns: i64,
        session: Option<&PySessionSchedule>,
        bar_ns: Option<i64>,
    ) -> PyResult<PyExprStream> {
        const OP: &str = "ExprProgram.stream";
        let plan = self.series(OP)?;
        let inner = if plan.has_at() {
            match clock_from(OP, bar_stamp, tz_offset_ns, session, bar_ns)? {
                Some(clock) => Stream::bound(plan, clock),
                None => Err(EvalError::NoClock),
            }
        } else {
            Stream::new(plan)
        }
        .map_err(|e| eval_fail(OP, e))?;
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

    /// Feed one bar stamped `timestamp` (nanoseconds, as the stream's `bar_stamp` reads them);
    /// returns one value per output. The only way to feed a program that uses `at(...)`.
    fn update_at(
        &self,
        timestamp: i64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> PyResult<Vec<f64>> {
        let mut s = self.locked()?;
        let out = s
            .update_at(timestamp, &Bar { open, high, low, close, volume }, &[])
            .map_err(|e| eval_fail("ExprStream.update_at", e))?;
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

fn array1<'py>(py: Python<'py>, v: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, v)
}

fn array2<'py>(
    py: Python<'py>,
    v: Vec<f64>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    Array2::from_shape_vec((rows, cols), v)
        .map(|a| a.into_pyarray(py))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// The factor report of `factor` against `prices` (both `(n_dates, n_assets)`), as a dict of
/// numbers and per-date arrays; `raptorbt.expressions.factor_report` is its surface. Forward
/// returns exist only in here.
#[pyfunction]
#[pyo3(signature = (factor, prices, horizons, quantiles, lag, min_names))]
pub fn factor_report_native<'py>(
    py: Python<'py>,
    factor: PyReadonlyArray2<'py, f64>,
    prices: PyReadonlyArray2<'py, f64>,
    horizons: Vec<usize>,
    quantiles: usize,
    lag: usize,
    min_names: usize,
) -> PyResult<Bound<'py, PyDict>> {
    const OP: &str = "factor_report";
    let (n, m) = (factor.shape()[0], factor.shape()[1]);
    if prices.shape() != [n, m] {
        return Err(shape_fail(OP, "prices", prices.shape(), (n, m)));
    }
    let (f, p) = (panel_cow(&factor), panel_cow(&prices));
    let params = FactorParams { horizons, quantiles, lag, min_names };
    let r = py.detach(|| factor_report(&f, &p, n, m, &params)).map_err(|e| {
        // `invalid_input` for every refusal: the stats codes have no Python entry yet.
        let fail = Fail::new("invalid_input", e.to_string()).op(OP);
        PyErr::from(match &e {
            crate::stats::StatsError::InsufficientObservations { what, .. } => fail.param(what),
            _ => fail,
        })
    })?;
    let out = PyDict::new(py);
    out.set_item("no_costs", r.no_costs)?;
    out.set_item("n_dates", r.n_dates)?;
    out.set_item("n_assets", r.n_assets)?;
    out.set_item("lag", r.lag)?;
    out.set_item("quantiles", r.quantiles)?;
    out.set_item("min_names", r.min_names)?;
    out.set_item("names", array1(py, r.names))?;
    out.set_item("coverage", array1(py, r.coverage))?;
    out.set_item("autocorr", array1(py, r.autocorr))?;
    out.set_item("autocorr_mean", r.autocorr_mean)?;
    out.set_item("turnover", array1(py, r.turnover))?;
    out.set_item("turnover_by_date", array2(py, r.turnover_by_date, n, r.quantiles)?)?;
    let by_h = PyDict::new(py);
    for h in r.horizons {
        let d = PyDict::new(py);
        d.set_item("ic", array1(py, h.ic))?;
        d.set_item("n_scored", h.n_scored)?;
        d.set_item("ic_mean", h.mean_ic)?;
        d.set_item("ic_std", h.std_ic)?;
        d.set_item("hit_rate", h.hit_rate)?;
        d.set_item("mean_names", h.mean_names)?;
        d.set_item("t_naive", h.t_naive)?;
        d.set_item("t_hac", h.t_hac)?;
        d.set_item("hac_lags", h.hac_lags)?;
        d.set_item("quantile_returns", array1(py, h.quantile_returns))?;
        d.set_item(
            "quantile_returns_by_date",
            array2(py, h.quantile_returns_by_date, n, r.quantiles)?,
        )?;
        d.set_item("long_short", array1(py, h.long_short))?;
        d.set_item("long_short_mean", h.long_short_mean)?;
        d.set_item("long_short_t_hac", h.long_short_t_hac)?;
        d.set_item("long_short_n", h.long_short_n)?;
        by_h.set_item(h.horizon, d)?;
    }
    out.set_item("horizons", by_h)?;
    Ok(out)
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
    m.add_function(wrap_pyfunction!(factor_report_native, m)?)?;
    Ok(())
}
