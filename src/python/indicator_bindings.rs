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

use super::errors::Fail;
use crate::runtime::{CellCore, IndKind, IndSpec, IndicatorBank, Ohlc, Reading, SlotId};

/// Where an indicator's value comes from.
#[derive(Debug)]
enum State {
    /// Its own streaming core, fed by `update_bar` or by the runtime.
    Private(Box<Private>),
    /// A reader of a shared bank cell: the runtime feeds the cell and decides
    /// which value this reader may see.
    Linked { bank: Arc<IndicatorBank>, slot: SlotId },
}

/// A privately owned indicator: its core and what a runtime shows of it.
///
/// A runtime feeds it when it applies an event, before any strategy decides
/// the events of the same timestamp group, so a decision on another instrument
/// reads the value after the event. The strategy's own hooks for the event
/// that fed it read the same protocol a shared cell has (see
/// [`crate::runtime::indicators`]): the value before the bar until the
/// runtime reveals it, the value after emission `k` once it has revealed `k`.
/// The event being decided is the bank's selected event.
#[derive(Debug)]
struct Private {
    core: CellCore,
    /// The value after the last emission.
    cur: Reading,
    /// The value before the event that last fed it.
    prev: Reading,
    /// The values after all but the last emission of that event.
    hist: Vec<Reading>,
    /// The event that last fed it (the bank's number), or `NOT_FED`.
    touched: u64,
    /// Emissions of that event.
    emissions: u32,
    /// The last reveal: the event it was made in and how many emissions it showed.
    revealed: (u64, u32),
    /// The runtime whose selected event decides what a read shows. `None`
    /// for an indicator fed by hand: it always shows its latest value.
    bank: Option<Arc<IndicatorBank>>,
    /// Fed by several instruments' streams (registered without a symbol):
    /// every event of the group is its strategy's own, as for a bank cell
    /// fed by an instrument set.
    fleet: bool,
}

/// The bank's number of an event never fed (and of a hand-fed update).
const NOT_FED: u64 = 0;

impl Private {
    fn new(spec: IndSpec) -> Self {
        Self {
            core: CellCore::for_spec(spec),
            cur: Reading::UNINIT,
            prev: Reading::UNINIT,
            hist: Vec::new(),
            touched: 0,
            emissions: 0,
            revealed: (NOT_FED, 0),
            bank: None,
            fleet: false,
        }
    }

    fn feed(&mut self, event: u64, floor: u64, bar: Ohlc) {
        if event == NOT_FED || self.touched < floor {
            self.prev = self.cur;
            self.hist.clear();
            self.emissions = 0;
            self.touched = event;
        } else {
            self.hist.push(self.cur);
        }
        self.cur = self.core.update(bar);
        self.emissions += 1;
    }

    /// Emissions of the selected event the strategy has been shown, or `None`
    /// if the whole latest value is shown.
    fn visible(&self) -> Option<u32> {
        let bank = self.bank.as_ref()?;
        let selected = bank.event_mark();
        let own = if self.fleet { self.touched >= bank.floor() } else { self.touched == selected };
        if self.touched == NOT_FED || !own {
            return None;
        }
        Some(if self.revealed.0 == selected { self.revealed.1 } else { 0 })
    }

    fn shown(&self) -> Reading {
        match self.visible() {
            None => self.cur,
            Some(0) => self.prev,
            Some(k) if k >= self.emissions => self.cur,
            Some(k) => self.hist[k as usize - 1],
        }
    }
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
    fn private(spec: IndSpec, kind: &str) -> Self {
        Self {
            state: State::Private(Box::new(Private::new(spec))),
            spec,
            updates: 0,
            kind: kind.into(),
        }
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

    /// Mark an indicator fed by several instruments' streams: it is shown
    /// after the whole group at every decision. No effect when linked.
    pub fn set_fleet(&mut self) {
        if let State::Private(p) = &mut self.state {
            p.fleet = true;
        }
    }

    /// Show what the selected event of `bank` has revealed of this indicator.
    /// No effect when linked.
    pub fn attach(&mut self, bank: Arc<IndicatorBank>) {
        if let State::Private(p) = &mut self.state {
            p.bank = Some(bank);
        }
    }

    /// The runtime applies one emission of the selected event: feed the core.
    /// Counts as an `update_bar` call. A linked indicator is fed by the bank.
    pub fn feed(&mut self, bar: Ohlc) {
        if let State::Private(p) = &mut self.state {
            self.updates += 1;
            let (event, floor) =
                p.bank.as_ref().map_or((NOT_FED, 0), |b| (b.event_mark(), b.floor()));
            p.feed(event, floor, bar);
        }
    }

    /// The strategy has processed emission `emission` of the selected event's
    /// stream (`u32::MAX` for all of them): show the value after it.
    pub fn reveal(&mut self, emission: u32) {
        if let State::Private(p) = &mut self.state {
            if let Some(bank) = &p.bank {
                p.revealed = (bank.event_mark(), emission);
            }
        }
    }

    /// Replace the private core with a read of `slot` in `bank`.
    pub fn link(&mut self, bank: Arc<IndicatorBank>, slot: SlotId) {
        self.state = State::Linked { bank, slot };
    }
}

/// Refuse a period below one, as the vectorised indicators do. The streaming
/// cores clamp a zero period to one, which would make `Indicator.sma(0)` an
/// SMA of one bar that reports itself initialised at once.
fn period_of(operation: &str, name: &str, value: i64) -> PyResult<usize> {
    if value < 1 {
        return Err(Fail::new(
            "invalid_parameter",
            format!("Invalid parameter: {operation} {name} must be > 0, got {value}"),
        )
        .op(operation)
        .param(name)
        .value(value)
        .into());
    }
    Ok(value as usize)
}

static BANDS: pyo3::sync::PyOnceLock<Py<PyAny>> = pyo3::sync::PyOnceLock::new();

/// `raptorbt.BollingerBands(upper, middle, lower)`: one order and names for
/// the array and the streaming form alike.
pub(crate) fn bollinger_bands_obj<'py, T: IntoPyObject<'py>>(
    py: Python<'py>,
    upper: T,
    middle: T,
    lower: T,
) -> PyResult<Py<PyAny>> {
    let class = BANDS.get_or_try_init(py, || {
        Ok::<_, PyErr>(py.import("raptorbt._bands")?.getattr("BollingerBands")?.unbind())
    })?;
    Ok(class.bind(py).call1((upper, middle, lower))?.unbind())
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
    if kind == IndKind::Bollinger {
        // The bank holds (middle, upper, lower); readers see the named order.
        return bollinger_bands_obj(py, b, a, c).map(Some);
    }
    match kind.width() {
        1 => a.into_py_any(py).map(Some),
        2 => (a, b).into_py_any(py).map(Some),
        _ => (a, b, c).into_py_any(py).map(Some),
    }
}

#[pymethods]
impl PyIndicator {
    #[staticmethod]
    fn sma(period: i64) -> PyResult<Self> {
        let period = period_of("Indicator.sma", "period", period)?;
        Ok(Self::private(IndSpec::Sma(period), "sma"))
    }

    #[staticmethod]
    fn ema(period: i64) -> PyResult<Self> {
        let period = period_of("Indicator.ema", "period", period)?;
        Ok(Self::private(IndSpec::Ema(period), "ema"))
    }

    /// Wilder-smoothed moving average (alpha = 1/period).
    #[staticmethod]
    fn wilder_ma(period: i64) -> PyResult<Self> {
        let period = period_of("Indicator.wilder_ma", "period", period)?;
        Ok(Self::private(IndSpec::WilderMa(period), "wilder_ma"))
    }

    #[staticmethod]
    fn wma(period: i64) -> PyResult<Self> {
        let period = period_of("Indicator.wma", "period", period)?;
        Ok(Self::private(IndSpec::Wma(period), "wma"))
    }

    /// Rate of change over `period` bars, in percent.
    #[staticmethod]
    fn roc(period: i64) -> PyResult<Self> {
        let period = period_of("Indicator.roc", "period", period)?;
        Ok(Self::private(IndSpec::Roc(period), "roc"))
    }

    /// Rolling population standard deviation.
    #[staticmethod]
    fn stddev(period: i64) -> PyResult<Self> {
        let period = period_of("Indicator.stddev", "period", period)?;
        Ok(Self::private(IndSpec::StdDev(period), "stddev"))
    }

    #[staticmethod]
    fn rsi(period: i64) -> PyResult<Self> {
        let period = period_of("Indicator.rsi", "period", period)?;
        Ok(Self::private(IndSpec::Rsi(period), "rsi"))
    }

    #[staticmethod]
    fn atr(period: i64) -> PyResult<Self> {
        let period = period_of("Indicator.atr", "period", period)?;
        Ok(Self::private(IndSpec::Atr(period), "atr"))
    }

    /// Highest-high / lowest-low channel; value is `(upper, lower)`.
    #[staticmethod]
    fn donchian(period: i64) -> PyResult<Self> {
        let period = period_of("Indicator.donchian", "period", period)?;
        Ok(Self::private(IndSpec::Donchian(period), "donchian"))
    }

    /// Bollinger bands; value is `BollingerBands(upper, middle, lower)`.
    #[staticmethod]
    #[pyo3(signature = (period, k=2.0))]
    fn bollinger(period: i64, k: f64) -> PyResult<Self> {
        let period = period_of("Indicator.bollinger", "period", period)?;
        if k.is_nan() || k <= 0.0 {
            return Err(Fail::new(
                "invalid_parameter",
                format!("Invalid parameter: Indicator.bollinger k must be > 0, got {k}"),
            )
            .op("Indicator.bollinger")
            .param("k")
            .value(k)
            .into());
        }
        Ok(Self::private(IndSpec::Bollinger { period, k }, "bollinger"))
    }

    /// MACD; value is `(macd, signal, histogram)`.
    #[staticmethod]
    #[pyo3(signature = (fast=12, slow=26, signal=9))]
    fn macd(fast: i64, slow: i64, signal: i64) -> PyResult<Self> {
        let fast = period_of("Indicator.macd", "fast", fast)?;
        let slow = period_of("Indicator.macd", "slow", slow)?;
        let signal = period_of("Indicator.macd", "signal", signal)?;
        if fast >= slow {
            return Err(Fail::new(
                "invalid_parameter",
                format!(
                    "Invalid parameter: Indicator.macd fast must be < slow, got fast={fast}, slow={slow}"
                ),
            )
            .op("Indicator.macd")
            .param("fast")
            .value(fast)
            .into());
        }
        Ok(Self::private(IndSpec::Macd { fast, slow, signal }, "macd"))
    }

    /// Feed one bar. Price-series cores consume the close; bar cores (atr,
    /// donchian) consume high/low/close. Returns the new value, if warm.
    fn update_bar(
        &mut self,
        py: Python<'_>,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
    ) -> PyResult<Option<Py<PyAny>>> {
        let State::Private(p) = &mut self.state else {
            return Err(linked_error("update_bar"));
        };
        self.updates += 1;
        p.feed(NOT_FED, 0, Ohlc { open, high, low, close });
        reading_value(py, self.spec.kind(), p.cur)
    }

    /// Latest value (`None` during warmup).
    #[getter]
    fn value(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let reading = match &self.state {
            State::Private(p) => p.shown(),
            State::Linked { bank, slot } => bank.read(*slot),
        };
        reading_value(py, self.spec.kind(), reading)
    }

    /// Whether the warmup period has completed.
    #[getter]
    fn initialized(&self) -> bool {
        match &self.state {
            State::Private(p) => p.shown().initialized,
            State::Linked { bank, slot } => bank.read(*slot).initialized,
        }
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
            State::Private(p) => match p.visible() {
                None => self.updates,
                Some(k) => self.updates - u64::from(p.emissions) + u64::from(k.min(p.emissions)),
            },
            State::Linked { bank, slot } => bank.updates_seen(*slot),
        }
    }

    /// Forget all state.
    fn reset(&mut self) -> PyResult<()> {
        let State::Private(p) = &mut self.state else {
            return Err(linked_error("reset"));
        };
        let (fleet, bank) = (p.fleet, p.bank.take());
        **p = Private::new(self.spec);
        (p.fleet, p.bank) = (fleet, bank);
        self.updates = 0;
        Ok(())
    }

    fn __repr__(&self) -> String {
        let linked = if self.is_linked() { ", linked" } else { "" };
        format!("Indicator(kind={}{linked})", self.kind)
    }
}
