//! Python bindings for trading-session calendars.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::errors::Fail;
use crate::core::session_schedule::{day_from_yyyymmdd, SessionSchedule};

/// Python-exposed trading-session calendar.
#[pyclass(name = "SessionSchedule")]
#[derive(Debug, Clone)]
pub struct PySessionSchedule {
    pub(crate) inner: SessionSchedule,
}

const WEEKDAYS: [&str; 7] = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"];

fn weekday_index(key: &Bound<'_, PyAny>) -> PyResult<usize> {
    if let Ok(n) = key.extract::<usize>() {
        if n < 7 {
            return Ok(n);
        }
    } else if let Ok(name) = key.extract::<String>() {
        if let Some(i) = WEEKDAYS.iter().position(|d| name.eq_ignore_ascii_case(d)) {
            return Ok(i);
        }
    }
    Err(Fail::new(
        "invalid_session_schedule",
        format!("windows key {key} is not a weekday: use 'mon'..'sun' or 0 (Monday) to 6 (Sunday)"),
    )
    .op("SessionSchedule")
    .param("windows")
    .valid(WEEKDAYS)
    .into())
}

fn value_err(e: crate::core::RaptorError) -> PyErr {
    Fail::new("invalid_session_schedule", e.to_string()).op("SessionSchedule").into()
}

#[pymethods]
impl PySessionSchedule {
    /// Build a schedule from per-weekday windows.
    ///
    /// `windows` maps a weekday (`"mon"`..`"sun"`, or 0 = Monday to 6 =
    /// Sunday) to a list of `(open_minute, close_minute)` local windows;
    /// two windows on a day give a lunch break. A weekday left out does not
    /// trade. `holidays` are local `yyyymmdd` ints. `squareoff_minute` is a
    /// local minute from midnight at which intraday positions are closed.
    #[new]
    #[pyo3(signature = (windows, tz_offset_ns=0, holidays=None, squareoff_minute=None))]
    fn new(
        windows: &Bound<'_, PyDict>,
        tz_offset_ns: i64,
        holidays: Option<Vec<i32>>,
        squareoff_minute: Option<u32>,
    ) -> PyResult<Self> {
        let mut by_day: [Vec<(u32, u32)>; 7] = Default::default();
        for (key, value) in windows.iter() {
            by_day[weekday_index(&key)?] = value.extract::<Vec<(u32, u32)>>().map_err(|_| {
                Fail::new(
                    "invalid_session_schedule",
                    format!("windows[{key}] must be a list of (open, close) minutes"),
                )
                .op("SessionSchedule")
                .param("windows")
            })?;
        }
        let inner = SessionSchedule::new(
            tz_offset_ns,
            by_day,
            holidays.unwrap_or_default(),
            squareoff_minute,
        )
        .map_err(value_err)?;
        Ok(Self { inner })
    }

    /// NSE cash equities: Mon-Fri 09:15-15:30 IST, square-off 15:20.
    #[staticmethod]
    #[pyo3(signature = (holidays=None, squareoff_minute=Some(920)))]
    fn nse_equity(holidays: Option<Vec<i32>>, squareoff_minute: Option<u32>) -> PyResult<Self> {
        Self::preset(SessionSchedule::nse_equity(), holidays, squareoff_minute)
    }

    /// NSE futures and options: Mon-Fri 09:15-15:30 IST, square-off 15:20.
    #[staticmethod]
    #[pyo3(signature = (holidays=None, squareoff_minute=Some(920)))]
    fn nse_fo(holidays: Option<Vec<i32>>, squareoff_minute: Option<u32>) -> PyResult<Self> {
        Self::preset(SessionSchedule::nse_fo(), holidays, squareoff_minute)
    }

    /// BSE cash equities: Mon-Fri 09:15-15:30 IST, square-off 15:20.
    #[staticmethod]
    #[pyo3(signature = (holidays=None, squareoff_minute=Some(920)))]
    fn bse_equity(holidays: Option<Vec<i32>>, squareoff_minute: Option<u32>) -> PyResult<Self> {
        Self::preset(SessionSchedule::bse_equity(), holidays, squareoff_minute)
    }

    /// A copy whose holiday set is exactly `dates` (`yyyymmdd` ints).
    fn with_holidays(&self, dates: Vec<i32>) -> PyResult<Self> {
        Ok(Self { inner: self.inner.clone().with_holidays(dates).map_err(value_err)? })
    }

    /// A copy with another square-off minute; `None` disables square-off.
    #[pyo3(signature = (squareoff_minute))]
    fn with_squareoff_minute(&self, squareoff_minute: Option<u32>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_squareoff_minute(squareoff_minute).map_err(value_err)?,
        })
    }

    /// Whether `ts_ns` is inside a trading window (`open <= t < close`).
    fn is_open(&self, ts_ns: i64) -> bool {
        self.inner.is_open(ts_ns)
    }

    /// `yyyymmdd` of the trading day `ts_ns` falls on, or None on a weekend
    /// or holiday. Time of day is ignored.
    fn session_date(&self, ts_ns: i64) -> Option<i32> {
        self.inner.session_date(ts_ns)
    }

    /// UTC nanoseconds of the last close on a `yyyymmdd` date; None if that
    /// date is not a trading day.
    fn session_close_ns(&self, date: i32) -> PyResult<Option<i64>> {
        if day_from_yyyymmdd(date).is_none() {
            return Err(Fail::new(
                "invalid_session_date",
                format!("{date} is not a yyyymmdd date"),
            )
            .op("SessionSchedule.session_close_ns")
            .param("date")
            .value(date)
            .into());
        }
        Ok(self.inner.session_close_ns(date))
    }

    /// The first window open strictly after `ts_ns`, or None.
    fn next_open_ns(&self, ts_ns: i64) -> Option<i64> {
        self.inner.next_open_ns(ts_ns)
    }

    #[getter]
    fn tz_offset_ns(&self) -> i64 {
        self.inner.tz_offset_ns()
    }

    #[getter]
    fn squareoff_minute(&self) -> Option<u32> {
        self.inner.squareoff_minute()
    }

    /// Sorted holiday dates as `yyyymmdd` ints.
    #[getter]
    fn holidays(&self) -> Vec<i32> {
        self.inner.holidays().collect()
    }

    /// Windows keyed `"mon"`..`"sun"`; days that do not trade are omitted.
    #[getter]
    fn windows<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (weekday, name) in WEEKDAYS.iter().enumerate() {
            if !self.inner.windows(weekday).is_empty() {
                out.set_item(name, self.inner.windows(weekday).to_vec())?;
            }
        }
        Ok(out)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!(
            "SessionSchedule(tz_offset_ns={}, squareoff_minute={:?}, holidays={})",
            self.inner.tz_offset_ns(),
            self.inner.squareoff_minute(),
            self.inner.holidays().count()
        )
    }
}

impl PySessionSchedule {
    fn preset(
        base: SessionSchedule,
        holidays: Option<Vec<i32>>,
        squareoff_minute: Option<u32>,
    ) -> PyResult<Self> {
        let mut inner = base.with_squareoff_minute(squareoff_minute).map_err(value_err)?;
        if let Some(dates) = holidays {
            inner = inner.with_holidays(dates).map_err(value_err)?;
        }
        Ok(Self { inner })
    }
}
