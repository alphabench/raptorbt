//! Python bindings for bar aggregation.
//!
//! Exposes a streaming [`BarAggregator`] (drives the strategy runner's
//! multi-timeframe subscriptions) and two batch helpers: `aggregate_bars`
//! (bars → coarser bars) and `bars_from_ticks` (raw tick arrays → bars).

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::core::types::{OhlcvBar, TickData};
use crate::data::{builder_for_with, AggregationUnit, BarBuilder, BarSpec, SourceRecord};

use super::numpy_bridge::{numpy_to_vec_f64, numpy_to_vec_i64, vec_to_numpy_f64, vec_to_numpy_i64};

fn parse_spec(step: u32, unit: &str) -> PyResult<BarSpec> {
    let unit = AggregationUnit::parse(unit).map_err(|e| PyValueError::new_err(e.to_string()))?;
    BarSpec::new(step, unit).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn make_builder_with(
    step: u32,
    unit: &str,
    tz_offset_ns: i64,
    brick_size: f64,
) -> PyResult<Box<dyn BarBuilder + Send>> {
    let params = crate::data::BuilderParams { brick_size };
    builder_for_with(parse_spec(step, unit)?, tz_offset_ns, params)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Streaming bar aggregator.
///
/// Push source records (finer bars or trades) in ascending time order; a
/// completed bar is returned as `(timestamp, open, high, low, close,
/// volume)` when its boundary is crossed. Time bars are stamped with their
/// window-end timestamp, so a bar labeled `t` contains only data before `t`.
#[pyclass(name = "BarAggregator")]
pub struct PyBarAggregator {
    builder: Box<dyn BarBuilder + Send>,
    #[pyo3(get)]
    step: u32,
    #[pyo3(get)]
    unit: String,
}

type BarTuple = (i64, f64, f64, f64, f64, f64);

fn to_tuple(bar: OhlcvBar) -> BarTuple {
    (bar.timestamp, bar.open, bar.high, bar.low, bar.close, bar.volume)
}

#[pymethods]
impl PyBarAggregator {
    #[new]
    #[pyo3(signature = (step, unit, tz_offset_ns=0, brick_size=0.0))]
    fn new(step: u32, unit: &str, tz_offset_ns: i64, brick_size: f64) -> PyResult<Self> {
        Ok(Self {
            builder: make_builder_with(step, unit, tz_offset_ns, brick_size)?,
            step,
            unit: unit.to_string(),
        })
    }

    /// Push one bar; returns the completed coarser bar, if any.
    #[allow(clippy::too_many_arguments)]
    fn push_bar(
        &mut self,
        timestamp: i64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Option<BarTuple> {
        let rec = SourceRecord { timestamp, open, high, low, close, volume, signed_volume: 0.0 };
        self.builder.push(&rec).map(to_tuple)
    }

    /// Push one trade; returns the completed bar, if any.
    ///
    /// `signed_size` is buy-initiated minus sell-initiated volume, which the
    /// signed-flow units consume. `0.0` means unknown, and those units then
    /// classify direction by the tick rule.
    #[pyo3(signature = (timestamp, price, size, signed_size=0.0))]
    fn push_trade(
        &mut self,
        timestamp: i64,
        price: f64,
        size: f64,
        signed_size: f64,
    ) -> Option<BarTuple> {
        self.builder
            .push(&SourceRecord::signed_trade(timestamp, price, size, signed_size))
            .map(to_tuple)
    }

    /// A bar completed alongside the last push but held back.
    ///
    /// One large move completes several Renko bricks at once, and `push`
    /// returns only the first. Drain this after every push, or those bars
    /// are lost.
    fn next_pending(&mut self) -> Option<BarTuple> {
        self.builder.next_pending().map(to_tuple)
    }

    /// Emit any in-progress bar at end of data.
    fn flush(&mut self) -> Option<BarTuple> {
        self.builder.flush().map(to_tuple)
    }
}

type BarArrays<'py> = (
    &'py PyArray1<i64>,
    &'py PyArray1<f64>,
    &'py PyArray1<f64>,
    &'py PyArray1<f64>,
    &'py PyArray1<f64>,
    &'py PyArray1<f64>,
);

fn bars_to_arrays(py: Python<'_>, bars: Vec<OhlcvBar>) -> BarArrays<'_> {
    let n = bars.len();
    let mut ts = Vec::with_capacity(n);
    let (mut o, mut h, mut l, mut c, mut v) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for bar in bars {
        ts.push(bar.timestamp);
        o.push(bar.open);
        h.push(bar.high);
        l.push(bar.low);
        c.push(bar.close);
        v.push(bar.volume);
    }
    (
        vec_to_numpy_i64(py, ts),
        vec_to_numpy_f64(py, o),
        vec_to_numpy_f64(py, h),
        vec_to_numpy_f64(py, l),
        vec_to_numpy_f64(py, c),
        vec_to_numpy_f64(py, v),
    )
}

/// Aggregate bars into coarser bars.
///
/// Includes the final in-progress bar (flushed at end of data). Supported
/// units: time (`"ms"`/`"s"`/`"m"`/`"h"`/`"d"`/`"w"`), `"tick"`,
/// `"volume"`, `"value"`.
#[pyfunction]
#[pyo3(signature = (timestamps, open, high, low, close, volume, step, unit, tz_offset_ns=0, brick_size=0.0))]
#[allow(clippy::too_many_arguments)]
pub fn aggregate_bars<'py>(
    py: Python<'py>,
    timestamps: PyReadonlyArray1<i64>,
    open: PyReadonlyArray1<f64>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    volume: PyReadonlyArray1<f64>,
    step: u32,
    unit: &str,
    tz_offset_ns: i64,
    brick_size: f64,
) -> PyResult<BarArrays<'py>> {
    let ts = numpy_to_vec_i64(timestamps);
    let o = numpy_to_vec_f64(open);
    let h = numpy_to_vec_f64(high);
    let l = numpy_to_vec_f64(low);
    let c = numpy_to_vec_f64(close);
    let v = numpy_to_vec_f64(volume);
    let n = ts.len();
    if [o.len(), h.len(), l.len(), c.len(), v.len()].iter().any(|&len| len != n) {
        return Err(PyValueError::new_err("all input arrays must share one length"));
    }

    let mut builder = make_builder_with(step, unit, tz_offset_ns, brick_size)?;
    let mut out = Vec::new();
    for i in 0..n {
        let rec = SourceRecord {
            timestamp: ts[i],
            open: o[i],
            high: h[i],
            low: l[i],
            close: c[i],
            volume: v[i],
            signed_volume: 0.0,
        };
        if let Some(bar) = builder.push(&rec) {
            out.push(bar);
        }
        // Renko can complete several bricks from one record.
        while let Some(bar) = builder.next_pending() {
            out.push(bar);
        }
    }
    if let Some(bar) = builder.flush() {
        out.push(bar);
    }
    Ok(bars_to_arrays(py, out))
}

/// Build bars from raw tick arrays (`ltp` + buy/sell quantity deltas).
///
/// Ticks with a zero last-traded price are skipped, matching the tick
/// backtester's treatment of missing data.
#[pyfunction]
#[pyo3(signature = (timestamps, ltp, buy_qty_delta, sell_qty_delta, step, unit, tz_offset_ns=0, brick_size=0.0))]
#[allow(clippy::too_many_arguments)]
pub fn bars_from_ticks<'py>(
    py: Python<'py>,
    timestamps: PyReadonlyArray1<i64>,
    ltp: PyReadonlyArray1<f64>,
    buy_qty_delta: PyReadonlyArray1<f64>,
    sell_qty_delta: PyReadonlyArray1<f64>,
    step: u32,
    unit: &str,
    tz_offset_ns: i64,
    brick_size: f64,
) -> PyResult<BarArrays<'py>> {
    let ts = numpy_to_vec_i64(timestamps);
    let prices = numpy_to_vec_f64(ltp);
    let buys = numpy_to_vec_f64(buy_qty_delta);
    let sells = numpy_to_vec_f64(sell_qty_delta);
    let n = ts.len();
    if [prices.len(), buys.len(), sells.len()].iter().any(|&len| len != n) {
        return Err(PyValueError::new_err("all input arrays must share one length"));
    }

    // Route through the shared conversion so bar-building and the future
    // event feed agree on which ticks count as trades.
    let ticks = TickData {
        timestamps: ts,
        ltp: prices,
        bid: vec![0.0; n],
        ask: vec![0.0; n],
        buy_qty_delta: buys,
        sell_qty_delta: sells,
        oi: vec![0.0; n],
    };
    let events = crate::data::tick_data_to_events(&ticks, 0, 0, 1);

    let mut builder = make_builder_with(step, unit, tz_offset_ns, brick_size)?;
    let mut out = Vec::new();
    for event in events {
        if let crate::data::EventPayload::Trade(trade) = event.payload {
            let rec = SourceRecord::signed_trade(
                trade.timestamp,
                trade.price,
                trade.size,
                trade.signed_size,
            );
            if let Some(bar) = builder.push(&rec) {
                out.push(bar);
            }
            while let Some(bar) = builder.next_pending() {
                out.push(bar);
            }
        }
    }
    if let Some(bar) = builder.flush() {
        out.push(bar);
    }
    Ok(bars_to_arrays(py, out))
}
