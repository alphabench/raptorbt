//! Python bindings for streaming indicators.
//!
//! One class, named constructors — `Indicator.sma(14)`, `.rsi(14)`,
//! `.bollinger(20, 2.0)`, ... Each instance consumes one record per
//! `update_bar` call and exposes `value` / `initialized`. Values match the
//! batch array functions exactly (equivalence-tested in Rust).

use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;

use crate::indicators::streaming::{
    StreamingAtr, StreamingBollinger, StreamingDonchian, StreamingEma, StreamingIndicator,
    StreamingMacd, StreamingRoc, StreamingRsi, StreamingSma, StreamingStdDev, StreamingWma,
};
use crate::runtime::{IndKind, IndSpec, IndicatorBank, Reading, SlotId};

#[derive(Debug)]
enum Core {
    Sma(StreamingSma),
    Ema(StreamingEma),
    Wilder(StreamingEma),
    Wma(StreamingWma),
    Roc(StreamingRoc),
    StdDev(StreamingStdDev),
    Rsi(StreamingRsi),
    Atr(StreamingAtr),
    Donchian(StreamingDonchian),
    Bollinger(StreamingBollinger),
    Macd(StreamingMacd),
}

/// Where an indicator's value comes from.
#[derive(Debug)]
enum State {
    /// Its own streaming core, fed by `update_bar`.
    Private(Box<Core>),
    /// A reader of a shared bank cell: the runtime feeds the cell and decides
    /// which value this reader may see.
    Linked { bank: Arc<IndicatorBank>, slot: SlotId },
}

/// Streaming indicator with batch-identical values.
#[pyclass(name = "Indicator")]
#[derive(Debug)]
pub struct PyIndicator {
    state: State,
    spec: IndSpec,
    /// `update_bar` calls so far; a linked indicator is never fed by hand.
    updates: u64,
    #[pyo3(get)]
    kind: String,
}

impl PyIndicator {
    fn private(core: Core, spec: IndSpec, kind: &str) -> Self {
        Self { state: State::Private(Box::new(core)), spec, updates: 0, kind: kind.into() }
    }

    /// What a bank needs to key this indicator.
    pub fn spec(&self) -> IndSpec {
        self.spec
    }

    /// Number of `update_bar` calls made on it.
    pub fn updates(&self) -> u64 {
        self.updates
    }

    pub fn is_linked(&self) -> bool {
        matches!(self.state, State::Linked { .. })
    }

    /// Replace the private core with a read of `slot` in `bank`.
    pub fn link(&mut self, bank: Arc<IndicatorBank>, slot: SlotId) {
        self.state = State::Linked { bank, slot };
    }
}

fn linked_error(what: &str) -> PyErr {
    PyRuntimeError::new_err(format!(
        "this indicator is linked to a shared runtime and updated by it: {what} is not available; \
         read value/initialized instead, or run with share=False for a privately owned indicator"
    ))
}

/// The value a reading shows, shaped like the private core's `value`.
fn reading_value(py: Python<'_>, kind: IndKind, r: Reading) -> PyResult<Option<Py<PyAny>>> {
    if !r.initialized {
        return Ok(None);
    }
    let [a, b, c] = r.values;
    match kind.width() {
        1 => a.into_py_any(py).map(Some),
        2 => (a, b).into_py_any(py).map(Some),
        _ => (a, b, c).into_py_any(py).map(Some),
    }
}

fn scalar(py: Python<'_>, v: Option<f64>) -> PyResult<Option<Py<PyAny>>> {
    v.map(|x| x.into_py_any(py)).transpose()
}

#[pymethods]
impl PyIndicator {
    #[staticmethod]
    fn sma(period: usize) -> Self {
        Self::private(Core::Sma(StreamingSma::new(period)), IndSpec::Sma(period), "sma")
    }

    #[staticmethod]
    fn ema(period: usize) -> Self {
        Self::private(Core::Ema(StreamingEma::new(period)), IndSpec::Ema(period), "ema")
    }

    /// Wilder-smoothed moving average (alpha = 1/period).
    #[staticmethod]
    fn wilder_ma(period: usize) -> Self {
        Self::private(
            Core::Wilder(StreamingEma::wilder(period)),
            IndSpec::WilderMa(period),
            "wilder_ma",
        )
    }

    #[staticmethod]
    fn wma(period: usize) -> Self {
        Self::private(Core::Wma(StreamingWma::new(period)), IndSpec::Wma(period), "wma")
    }

    /// Rate of change over `period` bars, in percent.
    #[staticmethod]
    fn roc(period: usize) -> Self {
        Self::private(Core::Roc(StreamingRoc::new(period)), IndSpec::Roc(period), "roc")
    }

    /// Rolling population standard deviation.
    #[staticmethod]
    fn stddev(period: usize) -> Self {
        Self::private(Core::StdDev(StreamingStdDev::new(period)), IndSpec::StdDev(period), "stddev")
    }

    #[staticmethod]
    fn rsi(period: usize) -> Self {
        Self::private(Core::Rsi(StreamingRsi::new(period)), IndSpec::Rsi(period), "rsi")
    }

    #[staticmethod]
    fn atr(period: usize) -> Self {
        Self::private(Core::Atr(StreamingAtr::new(period)), IndSpec::Atr(period), "atr")
    }

    /// Highest-high / lowest-low channel; value is `(upper, lower)`.
    #[staticmethod]
    fn donchian(period: usize) -> Self {
        Self::private(
            Core::Donchian(StreamingDonchian::new(period)),
            IndSpec::Donchian(period),
            "donchian",
        )
    }

    /// Bollinger bands; value is `(middle, upper, lower)`.
    #[staticmethod]
    #[pyo3(signature = (period, k=2.0))]
    fn bollinger(period: usize, k: f64) -> Self {
        Self::private(
            Core::Bollinger(StreamingBollinger::new(period, k)),
            IndSpec::Bollinger { period, k },
            "bollinger",
        )
    }

    /// MACD; value is `(macd, signal, histogram)`.
    #[staticmethod]
    #[pyo3(signature = (fast=12, slow=26, signal=9))]
    fn macd(fast: usize, slow: usize, signal: usize) -> Self {
        Self::private(
            Core::Macd(StreamingMacd::new(fast, slow, signal)),
            IndSpec::Macd { fast, slow, signal },
            "macd",
        )
    }

    /// Feed one bar. Price-series cores consume the close; bar cores (atr,
    /// donchian) consume high/low/close. Returns the new value, if warm.
    fn update_bar(
        &mut self,
        py: Python<'_>,
        _open: f64,
        high: f64,
        low: f64,
        close: f64,
    ) -> PyResult<Option<Py<PyAny>>> {
        let State::Private(core) = &mut self.state else {
            return Err(linked_error("update_bar"));
        };
        self.updates += 1;
        match &mut **core {
            Core::Sma(c) => scalar(py, c.update(close)),
            Core::Ema(c) | Core::Wilder(c) => scalar(py, c.update(close)),
            Core::Wma(c) => scalar(py, c.update(close)),
            Core::Roc(c) => scalar(py, c.update(close)),
            Core::StdDev(c) => scalar(py, c.update(close)),
            Core::Rsi(c) => scalar(py, c.update(close)),
            Core::Atr(c) => scalar(py, c.update_bar(high, low, close)),
            Core::Donchian(c) => c.update_bar(high, low).map(|v| v.into_py_any(py)).transpose(),
            Core::Bollinger(c) => c.update(close).map(|v| v.into_py_any(py)).transpose(),
            Core::Macd(c) => c.update(close).map(|v| v.into_py_any(py)).transpose(),
        }
    }

    /// Latest value (`None` during warmup).
    #[getter]
    fn value(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let core = match &self.state {
            State::Private(core) => core,
            State::Linked { bank, slot } => {
                return reading_value(py, self.spec.kind(), bank.read(*slot));
            }
        };
        match &**core {
            Core::Sma(c) => scalar(py, c.value()),
            Core::Ema(c) | Core::Wilder(c) => scalar(py, c.value()),
            Core::Wma(c) => scalar(py, c.value()),
            Core::Roc(c) => scalar(py, c.value()),
            Core::StdDev(c) => scalar(py, c.value()),
            Core::Rsi(c) => scalar(py, c.value()),
            Core::Atr(c) => scalar(py, c.value()),
            Core::Donchian(c) => c.value().map(|v| v.into_py_any(py)).transpose(),
            Core::Bollinger(c) => c.value().map(|v| v.into_py_any(py)).transpose(),
            Core::Macd(c) => c.value().map(|v| v.into_py_any(py)).transpose(),
        }
    }

    /// Whether the warmup period has completed.
    #[getter]
    fn initialized(&self, py: Python<'_>) -> PyResult<bool> {
        if let State::Linked { bank, slot } = &self.state {
            return Ok(bank.read(*slot).initialized);
        }
        Ok(self.value(py)?.is_some())
    }

    /// Whether a shared runtime owns and updates this indicator.
    #[getter]
    fn linked(&self) -> bool {
        self.is_linked()
    }

    /// How many bars it has been fed: `update_bar` calls, or for a linked
    /// indicator the bars of its shared cell its strategy has been shown.
    #[getter(update_count)]
    fn update_count(&self) -> u64 {
        match &self.state {
            State::Private(_) => self.updates,
            State::Linked { bank, slot } => bank.updates_seen(*slot),
        }
    }

    /// Forget all state.
    fn reset(&mut self) -> PyResult<()> {
        let State::Private(core) = &mut self.state else {
            return Err(linked_error("reset"));
        };
        match &mut **core {
            Core::Sma(c) => c.reset(),
            Core::Ema(c) | Core::Wilder(c) => c.reset(),
            Core::Wma(c) => c.reset(),
            Core::Roc(c) => c.reset(),
            Core::StdDev(c) => c.reset(),
            Core::Rsi(c) => c.reset(),
            Core::Atr(c) => c.reset(),
            Core::Donchian(c) => c.reset(),
            Core::Bollinger(c) => c.reset(),
            Core::Macd(c) => c.reset(),
        }
        self.updates = 0;
        Ok(())
    }

    fn __repr__(&self) -> String {
        let linked = if self.is_linked() { ", linked" } else { "" };
        format!("Indicator(kind={}{linked})", self.kind)
    }
}
