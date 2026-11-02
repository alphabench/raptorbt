//! Python bindings for resampling: the streaming `Resampler` and the batch
//! `resample_bars` / `resample_trades`, on the engine's time-window bar builder.
//!
//! One builder serves them all (`data::aggregate_resample`); nothing is computed here.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::errors::Fail;
use super::numpy_bridge::{
    numpy_to_vec_f64, numpy_to_vec_i64, vec_to_numpy_bool, vec_to_numpy_f64, vec_to_numpy_i64,
};
use crate::data::aggregate::{SourceRecord, TimeBarBuilder};
use crate::data::{CloseOn, Closed, ClosedBy, Gaps, ResampleError, ResampleOptions, ResampledBar};

fn code(e: &ResampleError) -> &'static str {
    match e {
        ResampleError::Late { .. } => "decreasing_timestamps",
        ResampleError::Invalid(_) | ResampleError::WrongInput(_) => "invalid_parameter",
        _ => "invalid_data",
    }
}

fn fail(e: ResampleError) -> PyErr {
    Fail::new(code(&e), e.to_string()).skill("timeframes").into()
}

fn fail_at((row, e): (usize, ResampleError)) -> PyErr {
    Fail::new(code(&e), format!("row {row}: {e}")).skill("timeframes").into()
}

#[allow(clippy::too_many_arguments)]
fn options(
    width_ns: i64,
    source_ns: i64,
    anchor_ns: i64,
    tz_offset_ns: i64,
    closed: &str,
    gaps: &str,
    close_on: &str,
    min_source: u64,
    session_anchored: bool,
) -> PyResult<ResampleOptions> {
    Ok(ResampleOptions {
        width_ns,
        anchor_ns,
        tz_offset_ns,
        closed: Closed::parse(closed).map_err(fail)?,
        gaps: Gaps::parse(gaps).map_err(fail)?,
        close_on: CloseOn::parse(close_on).map_err(fail)?,
        source_ns,
        min_source,
        session_anchored,
    })
}

/// A closed bar and how it came to be. `ts` is the bucket's end: nothing inside the bar
/// is known before it.
#[pyclass(frozen, name = "ResampledBar")]
pub struct PyResampledBar {
    bar: ResampledBar,
    width_ns: i64,
    stream: Option<String>,
}

#[pymethods]
impl PyResampledBar {
    #[getter]
    fn stream(&self) -> Option<&str> {
        self.stream.as_deref()
    }
    #[getter]
    fn ts(&self) -> i64 {
        self.bar.bar.timestamp
    }
    #[getter]
    fn open(&self) -> f64 {
        self.bar.bar.open
    }
    #[getter]
    fn high(&self) -> f64 {
        self.bar.bar.high
    }
    #[getter]
    fn low(&self) -> f64 {
        self.bar.bar.low
    }
    #[getter]
    fn close(&self) -> f64 {
        self.bar.bar.close
    }
    #[getter]
    fn volume(&self) -> f64 {
        self.bar.bar.volume
    }
    /// The bucket width in ns.
    #[getter]
    fn period(&self) -> i64 {
        self.width_ns
    }
    #[getter]
    fn kind(&self) -> &'static str {
        "time"
    }
    /// What closed the bar: "clock", "print", "carry", "session" or "flush".
    #[getter]
    fn closed_by(&self) -> &'static str {
        self.bar.closed_by.as_str()
    }
    /// How many source records built it; 0 for a carried bar.
    #[getter]
    fn n_source(&self) -> u64 {
        self.bar.n_source
    }
    /// False for a bar of resampled bars that saw fewer source bars than its width spans.
    #[getter]
    fn complete(&self) -> bool {
        self.bar.complete
    }
    /// `(ts, open, high, low, close, volume)`.
    fn as_tuple(&self) -> (i64, f64, f64, f64, f64, f64) {
        let b = &self.bar.bar;
        (b.timestamp, b.open, b.high, b.low, b.close, b.volume)
    }
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other.cast::<PyResampledBar>().is_ok_and(|o| {
            let o = o.get();
            self.bar == o.bar && self.width_ns == o.width_ns && self.stream == o.stream
        })
    }
    fn __repr__(&self) -> String {
        let b = &self.bar.bar;
        format!(
            "ResampledBar(ts={}, open={:?}, high={:?}, low={:?}, close={:?}, volume={:?}, \
             n_source={}, complete={}, closed_by={:?})",
            b.timestamp,
            b.open,
            b.high,
            b.low,
            b.close,
            b.volume,
            self.bar.n_source,
            self.bar.complete,
            self.bar.closed_by.as_str()
        )
    }
}

/// Bars from trades, or coarser bars from finer ones, on a grid you declare.
///
/// `width_ns` is the bucket width. `source_ns` is the width of the bars fed to `push_bar`
/// (0 for trades fed to `push_trade`); the width must be a multiple of it and the anchor must
/// sit on its grid. The grid has boundaries at `anchor_ns + k * width_ns` on the local clock
/// `tz_offset_ns` ahead of UTC (`IST_OFFSET_NS`), or, with `session_anchored`, restarts at
/// every `on_session(True, ts)`.
///
/// `closed` is which edge a record exactly on a boundary belongs to: `"left"` is
/// `[start, end)` (a boundary print opens the next bar; bars stamped at their open), `"right"`
/// is `(start, end]` (it closes the bar it ends; bars stamped at their close). `close_on` is
/// `"next_print"` (a bar closes when a later record arrives) or `"clock"` (as soon as
/// `on_clock` reaches its end). `gaps="carry"` emits a flat zero-volume bar at the last close
/// for every empty interval. `min_source` drops a bar built from fewer source records.
///
/// Every call returns the bars it closed, stamped at the bucket's end. A record earlier than
/// the open bucket, a non-finite price, a bar off the source grid or of another width are
/// refused, and the resampler is unchanged.
#[pyclass(name = "Resampler")]
pub struct PyResampler {
    inner: TimeBarBuilder,
    stream: Option<String>,
    buf: Vec<ResampledBar>,
}

impl PyResampler {
    fn run(
        &mut self,
        f: impl FnOnce(&mut TimeBarBuilder, &mut Vec<ResampledBar>) -> Result<(), ResampleError>,
    ) -> PyResult<Vec<PyResampledBar>> {
        self.buf.clear();
        let result = f(&mut self.inner, &mut self.buf);
        let width = self.inner.options().width_ns;
        let bars = self
            .buf
            .drain(..)
            .map(|bar| PyResampledBar { bar, width_ns: width, stream: self.stream.clone() })
            .collect();
        result.map(|()| bars).map_err(fail)
    }
}

#[pymethods]
impl PyResampler {
    #[new]
    #[pyo3(signature = (width_ns, *, source_ns=0, anchor_ns=0, tz_offset_ns=0, closed="left",
                        gaps="skip", close_on="next_print", min_source=0, session_anchored=false,
                        stream=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        width_ns: i64,
        source_ns: i64,
        anchor_ns: i64,
        tz_offset_ns: i64,
        closed: &str,
        gaps: &str,
        close_on: &str,
        min_source: u64,
        session_anchored: bool,
        stream: Option<String>,
    ) -> PyResult<Self> {
        let opts = options(
            width_ns,
            source_ns,
            anchor_ns,
            tz_offset_ns,
            closed,
            gaps,
            close_on,
            min_source,
            session_anchored,
        )?;
        Ok(Self { inner: TimeBarBuilder::resampler(opts).map_err(fail)?, stream, buf: Vec::new() })
    }

    #[getter]
    fn width_ns(&self) -> i64 {
        self.inner.options().width_ns
    }
    #[getter]
    fn source_ns(&self) -> i64 {
        self.inner.options().source_ns
    }
    #[getter]
    fn closed(&self) -> &'static str {
        self.inner.options().closed.as_str()
    }
    #[getter]
    fn gaps(&self) -> &'static str {
        self.inner.options().gaps.as_str()
    }
    #[getter]
    fn close_on(&self) -> &'static str {
        self.inner.options().close_on.as_str()
    }

    /// One trade; returns the bars it closed.
    fn push_trade(&mut self, ts: i64, price: f64, qty: f64) -> PyResult<Vec<PyResampledBar>> {
        self.run(|b, out| b.push_trade(ts, price, qty, out))
    }

    /// One source bar (stamped at its open for `closed="left"`, at its close for `"right"`);
    /// returns the bars it closed. `period` is the bar's width in ns when you want it
    /// checked against `source_ns` (0: not stated).
    #[pyo3(signature = (ts, open, high, low, close, volume=0.0, period=0))]
    #[allow(clippy::too_many_arguments)]
    fn push_bar(
        &mut self,
        ts: i64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
        period: i64,
    ) -> PyResult<Vec<PyResampledBar>> {
        self.run(|b, out| b.push_bar(ts, open, high, low, close, volume, period, out))
    }

    /// Time has reached `ts`: under `close_on="clock"`, closes a bar whose end has passed.
    fn on_clock(&mut self, ts: i64) -> PyResult<Vec<PyResampledBar>> {
        self.run(|b, out| b.on_clock(ts, out))
    }

    /// A session event at `ts` (`open` true for an open). Only a `session_anchored`
    /// resampler reads it as more than the clock.
    fn on_session(&mut self, open: bool, ts: i64) -> PyResult<Vec<PyResampledBar>> {
        self.run(|b, out| b.session(open, ts, out))
    }

    /// The instant the clock would next close the open bar, if any.
    fn next_close(&self) -> Option<i64> {
        self.inner.next_close()
    }

    /// The open bar as it stands, for a stream that has truly ended.
    fn flush(&mut self) -> PyResult<Vec<PyResampledBar>> {
        self.run(|b, out| {
            b.flush_resampled(out);
            Ok(())
        })
    }

    fn __repr__(&self) -> String {
        let o = self.inner.options();
        format!(
            "Resampler(width_ns={}, source_ns={}, closed={:?}, gaps={:?}, close_on={:?})",
            o.width_ns,
            o.source_ns,
            o.closed.as_str(),
            o.gaps.as_str(),
            o.close_on.as_str()
        )
    }
}

fn arrays<'py>(py: Python<'py>, bars: Vec<ResampledBar>) -> PyResult<Bound<'py, PyDict>> {
    let col = |f: fn(&ResampledBar) -> f64| bars.iter().map(f).collect::<Vec<f64>>();
    let d = PyDict::new(py);
    d.set_item("timestamp", vec_to_numpy_i64(py, bars.iter().map(|b| b.bar.timestamp).collect()))?;
    d.set_item("open", vec_to_numpy_f64(py, col(|b| b.bar.open)))?;
    d.set_item("high", vec_to_numpy_f64(py, col(|b| b.bar.high)))?;
    d.set_item("low", vec_to_numpy_f64(py, col(|b| b.bar.low)))?;
    d.set_item("close", vec_to_numpy_f64(py, col(|b| b.bar.close)))?;
    d.set_item("volume", vec_to_numpy_f64(py, col(|b| b.bar.volume)))?;
    d.set_item("n_source", vec_to_numpy_i64(py, bars.iter().map(|b| b.n_source as i64).collect()))?;
    d.set_item("complete", vec_to_numpy_bool(py, bars.iter().map(|b| b.complete).collect()))?;
    d.set_item(
        "carried",
        vec_to_numpy_bool(py, bars.iter().map(|b| b.closed_by == ClosedBy::Carry).collect()),
    )?;
    Ok(d)
}

fn same_length(n: usize, others: &[usize]) -> PyResult<()> {
    if others.iter().any(|&l| l != n) {
        return Err(Fail::new("length_mismatch", "all input arrays must share one length")
            .skill("timeframes")
            .into());
    }
    Ok(())
}

/// Resample bars into coarser bars; the batch form of [`Resampler`].
///
/// `source_ns` is the input bars' width (they are checked against the grid and each other);
/// `0` reads every row as a record (open, high, low, close, volume), as the plain
/// aggregator does. Returns a dict of arrays: `timestamp` (bucket ends), `open`, `high`,
/// `low`, `close`, `volume`, `n_source`, `complete` and `carried`. The final open bucket is
/// included when `flush` (default). The GIL is released while it runs.
#[pyfunction]
#[pyo3(signature = (timestamps, open, high, low, close, volume, width_ns, *, source_ns=0,
                    anchor_ns=0, tz_offset_ns=0, closed="left", gaps="skip", min_source=0,
                    flush=true))]
#[allow(clippy::too_many_arguments)]
pub fn resample_bars<'py>(
    py: Python<'py>,
    timestamps: PyReadonlyArray1<i64>,
    open: PyReadonlyArray1<f64>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    volume: PyReadonlyArray1<f64>,
    width_ns: i64,
    source_ns: i64,
    anchor_ns: i64,
    tz_offset_ns: i64,
    closed: &str,
    gaps: &str,
    min_source: u64,
    flush: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let ts = numpy_to_vec_i64(timestamps);
    let (o, h, l, c, v) = (
        numpy_to_vec_f64(open),
        numpy_to_vec_f64(high),
        numpy_to_vec_f64(low),
        numpy_to_vec_f64(close),
        numpy_to_vec_f64(volume),
    );
    same_length(ts.len(), &[o.len(), h.len(), l.len(), c.len(), v.len()])?;
    let opts = options(
        width_ns,
        source_ns,
        anchor_ns,
        tz_offset_ns,
        closed,
        gaps,
        "next_print",
        min_source,
        false,
    )?;
    let out = py.detach(|| -> Result<Vec<ResampledBar>, (usize, ResampleError)> {
        let mut b = TimeBarBuilder::resampler(opts).map_err(|e| (0, e))?;
        let mut out = Vec::new();
        for i in 0..ts.len() {
            let r = if source_ns > 0 {
                b.push_bar(ts[i], o[i], h[i], l[i], c[i], v[i], 0, &mut out)
            } else {
                let rec = SourceRecord {
                    timestamp: ts[i],
                    open: o[i],
                    high: h[i],
                    low: l[i],
                    close: c[i],
                    volume: v[i],
                    signed_volume: 0.0,
                };
                b.absorb(&rec, &mut out)
            };
            r.map_err(|e| (i, e))?;
        }
        if flush {
            b.flush_resampled(&mut out);
        }
        Ok(out)
    });
    arrays(py, out.map_err(fail_at)?)
}

/// Resample trades into bars: [`resample_bars`] for a `price` and `qty` per row.
#[pyfunction]
#[pyo3(signature = (timestamps, price, qty, width_ns, *, anchor_ns=0, tz_offset_ns=0,
                    closed="left", gaps="skip", min_source=0, flush=true))]
#[allow(clippy::too_many_arguments)]
pub fn resample_trades<'py>(
    py: Python<'py>,
    timestamps: PyReadonlyArray1<i64>,
    price: PyReadonlyArray1<f64>,
    qty: PyReadonlyArray1<f64>,
    width_ns: i64,
    anchor_ns: i64,
    tz_offset_ns: i64,
    closed: &str,
    gaps: &str,
    min_source: u64,
    flush: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let p = numpy_to_vec_f64(price);
    let ts = numpy_to_vec_i64(timestamps);
    let q = numpy_to_vec_f64(qty);
    same_length(ts.len(), &[p.len(), q.len()])?;
    let opts = options(
        width_ns,
        0,
        anchor_ns,
        tz_offset_ns,
        closed,
        gaps,
        "next_print",
        min_source,
        false,
    )?;
    let out = py.detach(|| -> Result<Vec<ResampledBar>, (usize, ResampleError)> {
        let mut b = TimeBarBuilder::resampler(opts).map_err(|e| (0, e))?;
        let mut out = Vec::new();
        for i in 0..ts.len() {
            b.absorb(&SourceRecord::trade(ts[i], p[i], q[i]), &mut out).map_err(|e| (i, e))?;
        }
        if flush {
            b.flush_resampled(&mut out);
        }
        Ok(out)
    });
    arrays(py, out.map_err(fail_at)?)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyResampler>()?;
    m.add_class::<PyResampledBar>()?;
    m.add_function(wrap_pyfunction!(resample_bars, m)?)?;
    m.add_function(wrap_pyfunction!(resample_trades, m)?)?;
    Ok(())
}
