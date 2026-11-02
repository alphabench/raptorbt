//! Python bindings for the research numerics (`raptorbt.research`): the deflated Sharpe
//! ratio, the stationary bootstrap, `without_best`, walk-forward windows and parameter
//! designs.
//!
//! The functions return plain dicts, which `python/raptorbt/research.py` consumes as they
//! are. Seeds are masked to 64 bits, so a Python int of any size or sign maps the same way.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyInt, PyList, PyTuple};

use crate::research::{
    deflated_sharpe as compute_deflated, stationary_bootstrap as compute_bootstrap,
    without_best as compute_without_best, Bootstrap, DeflatedSharpe, Design, Luck, ParamDraw,
    ParamSpace, WalkForward,
};

/// Seeds are masked to 64 bits; a Python int of any size or sign maps the same way.
fn seed_bits(seed: &Bound<'_, PyAny>) -> PyResult<u64> {
    let masked = seed.call_method1("__and__", (u64::MAX,))?;
    if masked.is_none() || !masked.is_instance_of::<PyInt>() {
        return Err(PyValueError::new_err("seed is an integer"));
    }
    masked.extract::<u64>()
}

#[pyfunction]
#[pyo3(signature = (sharpes, best, *, samples, skew=0.0, kurtosis=3.0, periods_per_year=252.0, luck="trials"))]
#[allow(clippy::too_many_arguments)]
fn deflated_sharpe<'py>(
    py: Python<'py>,
    sharpes: Vec<f64>,
    best: f64,
    samples: i64,
    skew: f64,
    kurtosis: f64,
    periods_per_year: f64,
    luck: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let luck = Luck::parse(luck)?;
    let d = PyDict::new(py);
    match compute_deflated(&sharpes, best, samples, skew, kurtosis, periods_per_year, luck) {
        DeflatedSharpe::Insufficient { trials } => {
            d.set_item("trials", trials)?;
            d.set_item("sr0", py.None())?;
            d.set_item("dsr", py.None())?;
            d.set_item("note", "needs >= 2 trials and 3 samples")?;
        }
        DeflatedSharpe::MomentsOutOfRange { trials, sr0_annualised } => {
            d.set_item("trials", trials)?;
            d.set_item("sr0", sr0_annualised)?;
            d.set_item("dsr", py.None())?;
            d.set_item("note", "moments out of range")?;
        }
        DeflatedSharpe::Computed {
            trials,
            luck,
            sr0_annualised,
            best_annualised,
            dsr,
            z,
            skew,
            kurtosis,
            samples,
        } => {
            d.set_item("trials", trials)?;
            d.set_item("luck", luck.as_str())?;
            d.set_item("sr0_annualised", sr0_annualised)?;
            d.set_item("best_annualised", best_annualised)?;
            d.set_item("dsr", dsr)?;
            d.set_item("z", z)?;
            d.set_item("skew", skew)?;
            d.set_item("kurtosis", kurtosis)?;
            d.set_item("samples", samples)?;
        }
    }
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (returns, *, draws=2000, block=None, seed, periods_per_year=252.0))]
fn stationary_bootstrap<'py>(
    py: Python<'py>,
    returns: Vec<f64>,
    draws: i64,
    block: Option<f64>,
    seed: &Bound<'py, PyAny>,
    periods_per_year: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let bits = seed_bits(seed)?;
    let draws = usize::try_from(draws.max(0)).unwrap_or(0);
    let d = PyDict::new(py);
    match compute_bootstrap(&returns, draws, block, bits, periods_per_year)? {
        Bootstrap::Insufficient { samples } => {
            d.set_item("samples", samples)?;
            d.set_item("draws", 0)?;
            d.set_item("note", "needs >= 2 samples and 1 draw")?;
        }
        Bootstrap::Drawn { samples, draws, block, p05, p50, p95, above_zero } => {
            d.set_item("samples", samples)?;
            d.set_item("draws", draws)?;
            d.set_item("block", block)?;
            d.set_item("seed", seed)?;
            let ar = PyDict::new(py);
            ar.set_item("p05", p05)?;
            ar.set_item("p50", p50)?;
            ar.set_item("p95", p95)?;
            d.set_item("annual_return", ar)?;
            d.set_item("above_zero", above_zero)?;
        }
    }
    Ok(d)
}

#[pyfunction]
fn without_best(returns: Vec<f64>, top: i64) -> PyResult<f64> {
    Ok(compute_without_best(&returns, top)?)
}

/// `[{"index", "train", "test"}]`, each day list a slice of `days` in order.
#[pyfunction]
#[pyo3(signature = (days, *, train, test, step=None))]
fn walk_forward_windows<'py>(
    py: Python<'py>,
    days: Vec<Py<PyAny>>,
    train: i64,
    test: i64,
    step: Option<i64>,
) -> PyResult<Bound<'py, PyList>> {
    let wf = WalkForward::new(days.len(), train, test, step)?;
    let out = PyList::empty(py);
    for w in wf.windows() {
        let d = PyDict::new(py);
        d.set_item("index", w.index)?;
        d.set_item("train", PyList::new(py, days[w.train].iter().map(|x| x.bind(py)))?)?;
        d.set_item("test", PyList::new(py, days[w.test].iter().map(|x| x.bind(py)))?)?;
        out.append(d)?;
    }
    Ok(out)
}

fn sorted_space<'py>(
    space: &Bound<'py, PyDict>,
) -> PyResult<Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>> {
    let mut items: Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)> = space.iter().collect();
    // Python's sorted(space): keys compare with `<`
    let mut err: Option<PyErr> = None;
    items.sort_by(|a, b| match a.0.lt(&b.0) {
        Ok(true) => std::cmp::Ordering::Less,
        Ok(false) => match b.0.lt(&a.0) {
            Ok(true) => std::cmp::Ordering::Greater,
            Ok(false) => std::cmp::Ordering::Equal,
            Err(e) => {
                err.get_or_insert(e);
                std::cmp::Ordering::Equal
            }
        },
        Err(e) => {
            err.get_or_insert(e);
            std::cmp::Ordering::Equal
        }
    });
    match err {
        Some(e) => Err(e),
        None => Ok(items),
    }
}

fn choices<'py>(v: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    v.try_iter()?.collect()
}

/// The full product of `space` (key → choices) in sorted key order, the last key varying
/// fastest, as a list of dicts.
#[pyfunction]
fn design_grid<'py>(py: Python<'py>, space: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyList>> {
    let items = sorted_space(space)?;
    let lists: Vec<Vec<Bound<'py, PyAny>>> =
        items.iter().map(|(_, v)| choices(v)).collect::<PyResult<_>>()?;
    let sizes: Vec<usize> = lists.iter().map(Vec::len).collect();
    let out = PyList::empty(py);
    for point in Design::grid(&sizes) {
        let d = PyDict::new(py);
        for (k, idx) in point.iter().enumerate() {
            d.set_item(&items[k].0, &lists[k][*idx])?;
        }
        out.append(d)?;
    }
    Ok(out)
}

fn is_number(v: &Bound<'_, PyAny>) -> bool {
    v.is_instance_of::<PyInt>() || v.is_instance_of::<PyFloat>()
}

/// `n` points drawn by SplitMix64 from `seed`: a `(lo, hi)` tuple of numbers is a range
/// (an integer when both bounds are), anything else is a list of choices.
#[pyfunction]
#[pyo3(signature = (space, n, *, seed))]
fn design_random<'py>(
    py: Python<'py>,
    space: &Bound<'py, PyDict>,
    n: i64,
    seed: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let bits = seed_bits(seed)?;
    let items = sorted_space(space)?;
    let mut spaces = Vec::with_capacity(items.len());
    let mut lists = Vec::with_capacity(items.len());
    for (_, v) in &items {
        let range = match v.cast::<PyTuple>() {
            Ok(t) if t.len() == 2 && t.iter().all(|x| is_number(&x)) => {
                let (lo, hi) = (t.get_item(0)?, t.get_item(1)?);
                if lo.is_instance_of::<PyInt>() && hi.is_instance_of::<PyInt>() {
                    Some(ParamSpace::IntRange { lo: lo.extract()?, hi: hi.extract()? })
                } else {
                    Some(ParamSpace::FloatRange { lo: lo.extract()?, hi: hi.extract()? })
                }
            }
            _ => None,
        };
        match range {
            Some(r) => {
                spaces.push(r);
                lists.push(Vec::new());
            }
            None => {
                let c = choices(v)?;
                spaces.push(ParamSpace::Choices(c.len()));
                lists.push(c);
            }
        }
    }
    let n = usize::try_from(n).map_err(|_| PyValueError::new_err("n is not negative"))?;
    let out = PyList::empty(py);
    for point in Design::random(&spaces, n, bits)? {
        let d = PyDict::new(py);
        for (k, draw) in point.iter().enumerate() {
            let key = &items[k].0;
            match *draw {
                ParamDraw::Choice(i) => d.set_item(key, &lists[k][i])?,
                ParamDraw::Int(v) => d.set_item(key, v)?,
                ParamDraw::Float(v) => d.set_item(key, v)?,
            }
        }
        out.append(d)?;
    }
    Ok(out)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(deflated_sharpe, m)?)?;
    m.add_function(wrap_pyfunction!(stationary_bootstrap, m)?)?;
    m.add_function(wrap_pyfunction!(without_best, m)?)?;
    m.add_function(wrap_pyfunction!(walk_forward_windows, m)?)?;
    m.add_function(wrap_pyfunction!(design_grid, m)?)?;
    m.add_function(wrap_pyfunction!(design_random, m)?)?;
    Ok(())
}
