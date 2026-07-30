//! Python bindings for streaming indicators.
//!
//! One class, named constructors — `Indicator.sma(14)`, `.rsi(14)`,
//! `.bollinger(20, 2.0)`, ... Each instance consumes one record per
//! `update_bar` call and exposes `value` / `initialized`. Values match the
//! batch array functions exactly (equivalence-tested in Rust).

use pyo3::prelude::*;

use crate::indicators::streaming::{
    StreamingAtr, StreamingBollinger, StreamingDonchian, StreamingEma, StreamingIndicator,
    StreamingMacd, StreamingRoc, StreamingRsi, StreamingSma, StreamingStdDev, StreamingWma,
};

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

/// Streaming indicator with batch-identical values.
#[pyclass(name = "Indicator")]
#[derive(Debug)]
pub struct PyIndicator {
    core: Core,
    #[pyo3(get)]
    kind: String,
}

fn scalar(py: Python<'_>, v: Option<f64>) -> Option<PyObject> {
    v.map(|x| x.into_py(py))
}

#[pymethods]
impl PyIndicator {
    #[staticmethod]
    fn sma(period: usize) -> Self {
        Self { core: Core::Sma(StreamingSma::new(period)), kind: "sma".into() }
    }

    #[staticmethod]
    fn ema(period: usize) -> Self {
        Self { core: Core::Ema(StreamingEma::new(period)), kind: "ema".into() }
    }

    /// Wilder-smoothed moving average (alpha = 1/period).
    #[staticmethod]
    fn wilder_ma(period: usize) -> Self {
        Self { core: Core::Wilder(StreamingEma::wilder(period)), kind: "wilder_ma".into() }
    }

    #[staticmethod]
    fn wma(period: usize) -> Self {
        Self { core: Core::Wma(StreamingWma::new(period)), kind: "wma".into() }
    }

    /// Rate of change over `period` bars, in percent.
    #[staticmethod]
    fn roc(period: usize) -> Self {
        Self { core: Core::Roc(StreamingRoc::new(period)), kind: "roc".into() }
    }

    /// Rolling population standard deviation.
    #[staticmethod]
    fn stddev(period: usize) -> Self {
        Self { core: Core::StdDev(StreamingStdDev::new(period)), kind: "stddev".into() }
    }

    #[staticmethod]
    fn rsi(period: usize) -> Self {
        Self { core: Core::Rsi(StreamingRsi::new(period)), kind: "rsi".into() }
    }

    #[staticmethod]
    fn atr(period: usize) -> Self {
        Self { core: Core::Atr(StreamingAtr::new(period)), kind: "atr".into() }
    }

    /// Highest-high / lowest-low channel; value is `(upper, lower)`.
    #[staticmethod]
    fn donchian(period: usize) -> Self {
        Self { core: Core::Donchian(StreamingDonchian::new(period)), kind: "donchian".into() }
    }

    /// Bollinger bands; value is `(middle, upper, lower)`.
    #[staticmethod]
    #[pyo3(signature = (period, k=2.0))]
    fn bollinger(period: usize, k: f64) -> Self {
        Self { core: Core::Bollinger(StreamingBollinger::new(period, k)), kind: "bollinger".into() }
    }

    /// MACD; value is `(macd, signal, histogram)`.
    #[staticmethod]
    #[pyo3(signature = (fast=12, slow=26, signal=9))]
    fn macd(fast: usize, slow: usize, signal: usize) -> Self {
        Self { core: Core::Macd(StreamingMacd::new(fast, slow, signal)), kind: "macd".into() }
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
    ) -> Option<PyObject> {
        match &mut self.core {
            Core::Sma(c) => scalar(py, c.update(close)),
            Core::Ema(c) | Core::Wilder(c) => scalar(py, c.update(close)),
            Core::Wma(c) => scalar(py, c.update(close)),
            Core::Roc(c) => scalar(py, c.update(close)),
            Core::StdDev(c) => scalar(py, c.update(close)),
            Core::Rsi(c) => scalar(py, c.update(close)),
            Core::Atr(c) => scalar(py, c.update_bar(high, low, close)),
            Core::Donchian(c) => c.update_bar(high, low).map(|v| v.into_py(py)),
            Core::Bollinger(c) => c.update(close).map(|v| v.into_py(py)),
            Core::Macd(c) => c.update(close).map(|v| v.into_py(py)),
        }
    }

    /// Latest value (`None` during warmup).
    #[getter]
    fn value(&self, py: Python<'_>) -> Option<PyObject> {
        match &self.core {
            Core::Sma(c) => scalar(py, c.value()),
            Core::Ema(c) | Core::Wilder(c) => scalar(py, c.value()),
            Core::Wma(c) => scalar(py, c.value()),
            Core::Roc(c) => scalar(py, c.value()),
            Core::StdDev(c) => scalar(py, c.value()),
            Core::Rsi(c) => scalar(py, c.value()),
            Core::Atr(c) => scalar(py, c.value()),
            Core::Donchian(c) => c.value().map(|v| v.into_py(py)),
            Core::Bollinger(c) => c.value().map(|v| v.into_py(py)),
            Core::Macd(c) => c.value().map(|v| v.into_py(py)),
        }
    }

    /// Whether the warmup period has completed.
    #[getter]
    fn initialized(&self, py: Python<'_>) -> bool {
        self.value(py).is_some()
    }

    /// Forget all state.
    fn reset(&mut self) {
        match &mut self.core {
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
    }

    fn __repr__(&self) -> String {
        format!("Indicator(kind={})", self.kind)
    }
}
