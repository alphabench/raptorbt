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
        let by_day = Self::weekday_windows(windows)?;
        let inner = SessionSchedule::new(
            tz_offset_ns,
            by_day,
            holidays.unwrap_or_default(),
            squareoff_minute,
        )
        .map_err(value_err)?;
        Ok(Self { inner })
    }

    /// NSE cash equities: Mon-Fri 09:15-15:30 IST. No square-off unless
    /// `squareoff_minute` is given.
    #[staticmethod]
    #[pyo3(signature = (holidays=None, squareoff_minute=None))]
    fn nse_equity(holidays: Option<Vec<i32>>, squareoff_minute: Option<u32>) -> PyResult<Self> {
        Self::preset(SessionSchedule::nse_equity(), holidays, squareoff_minute)
    }

    /// NSE equity derivatives: Mon-Fri 09:15-15:30 IST, 09:15-15:40 from
    /// 2026-08-03.
    #[staticmethod]
    #[pyo3(signature = (holidays=None, squareoff_minute=None))]
    fn nse_fo(holidays: Option<Vec<i32>>, squareoff_minute: Option<u32>) -> PyResult<Self> {
        Self::preset(SessionSchedule::nse_fo(), holidays, squareoff_minute)
    }

    /// BSE cash equities: Mon-Fri 09:15-15:30 IST.
    #[staticmethod]
    #[pyo3(signature = (holidays=None, squareoff_minute=None))]
    fn bse_equity(holidays: Option<Vec<i32>>, squareoff_minute: Option<u32>) -> PyResult<Self> {
        Self::preset(SessionSchedule::bse_equity(), holidays, squareoff_minute)
    }

    /// BSE equity derivatives: as `nse_fo`.
    #[staticmethod]
    #[pyo3(signature = (holidays=None, squareoff_minute=None))]
    fn bse_fo(holidays: Option<Vec<i32>>, squareoff_minute: Option<u32>) -> PyResult<Self> {
        Self::preset(SessionSchedule::bse_fo(), holidays, squareoff_minute)
    }

    /// MCX non-agricultural contracts: a day session 09:00-17:00 and an
    /// evening session 17:00-23:55 IST, 23:30 while the US is on daylight saving
    /// (from the Monday after the second Sunday of March to the Sunday that is
    /// the first of November). Covered from 2024-11-04, with no end.
    #[staticmethod]
    #[pyo3(signature = (holidays=None, squareoff_minute=None))]
    fn mcx(holidays: Option<Vec<i32>>, squareoff_minute: Option<u32>) -> PyResult<Self> {
        Self::preset(SessionSchedule::mcx(), holidays, squareoff_minute)
    }

    /// MCX internationally referenceable agricultural contracts (Cotton, Cotton
    /// Seed Oil / CPO, Kapas): 09:00-21:00 all year, covered as `mcx`.
    #[staticmethod]
    #[pyo3(signature = (holidays=None, squareoff_minute=None))]
    fn mcx_select_agri(
        holidays: Option<Vec<i32>>,
        squareoff_minute: Option<u32>,
    ) -> PyResult<Self> {
        Self::preset(SessionSchedule::mcx_select_agri(), holidays, squareoff_minute)
    }

    /// All other MCX agricultural contracts: 09:00-17:00 all year.
    #[staticmethod]
    #[pyo3(signature = (holidays=None, squareoff_minute=None))]
    fn mcx_other_agri(holidays: Option<Vec<i32>>, squareoff_minute: Option<u32>) -> PyResult<Self> {
        Self::preset(SessionSchedule::mcx_other_agri(), holidays, squareoff_minute)
    }

    /// NSE currency derivatives on cross-currency pairs: Mon-Fri 09:00-19:30 IST.
    #[staticmethod]
    #[pyo3(signature = (holidays=None, squareoff_minute=None))]
    fn cds_cross(holidays: Option<Vec<i32>>, squareoff_minute: Option<u32>) -> PyResult<Self> {
        Self::preset(SessionSchedule::cds_cross(), holidays, squareoff_minute)
    }

    /// NSE currency derivatives on INR pairs: Mon-Fri 09:00-17:00 IST.
    #[staticmethod]
    #[pyo3(signature = (holidays=None, squareoff_minute=None))]
    fn cds(holidays: Option<Vec<i32>>, squareoff_minute: Option<u32>) -> PyResult<Self> {
        Self::preset(SessionSchedule::cds(), holidays, squareoff_minute)
    }

    /// BSE currency derivatives: as `cds`.
    #[staticmethod]
    #[pyo3(signature = (holidays=None, squareoff_minute=None))]
    fn bcd(holidays: Option<Vec<i32>>, squareoff_minute: Option<u32>) -> PyResult<Self> {
        Self::preset(SessionSchedule::bcd(), holidays, squareoff_minute)
    }

    /// One venue's schedule from dated calendar documents (JSON text, later
    /// documents win where they overlap); see the `SessionSchedule`
    /// docstring for the format. A date no document covers is closed.
    #[staticmethod]
    fn from_calendars(venue: &str, documents: Vec<String>) -> PyResult<Self> {
        let docs: Vec<&str> = documents.iter().map(String::as_str).collect();
        SessionSchedule::from_calendars(venue, &docs).map(|inner| Self { inner }).map_err(|e| {
            Fail::new("invalid_calendar", e.to_string())
                .op("SessionSchedule.from_calendars")
                .param("documents")
                .into()
        })
    }

    /// A copy with other weekday windows from `from_date` to `to_date`
    /// (`yyyymmdd`, inclusive; `to_date=None` is open-ended). `windows` is
    /// keyed like the constructor's; a weekday left out does not trade in
    /// that period.
    #[pyo3(signature = (from_date, windows, to_date=None))]
    fn with_window_change(
        &self,
        from_date: i32,
        windows: &Bound<'_, PyDict>,
        to_date: Option<i32>,
    ) -> PyResult<Self> {
        let by_day = Self::weekday_windows(windows)?;
        Ok(Self {
            inner: self
                .inner
                .clone()
                .with_window_change(from_date, to_date, by_day)
                .map_err(value_err)?,
        })
    }

    /// A copy that trades `windows` (`[(open_minute, close_minute), ...]`) on
    /// the local date `date` (`yyyymmdd`) whatever the weekday or holiday
    /// set says: Muhurat trading, a special session, a half day. Empty
    /// windows close the date.
    fn with_session_override(&self, date: i32, windows: Vec<(u32, u32)>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.clone().with_session_override(date, windows).map_err(value_err)?,
        })
    }

    /// The windows of a local `yyyymmdd` date after overrides, holidays and
    /// dated changes; empty when it does not trade.
    fn windows_on(&self, date: i32) -> Vec<(u32, u32)> {
        self.inner.windows_on(date)
    }

    /// Trading minutes in a regular session (the longest weekday).
    #[getter]
    fn trading_minutes(&self) -> f64 {
        self.inner.trading_minutes()
    }

    /// The `(from, to)` `yyyymmdd` ranges the schedule answers for; empty
    /// means every date. A date outside them is closed.
    #[getter]
    fn coverage(&self) -> Vec<(i32, i32)> {
        self.inner.coverage()
    }

    /// Per-date overrides as `{yyyymmdd: [(open, close), ...]}`.
    #[getter]
    fn overrides(&self) -> std::collections::BTreeMap<i32, Vec<(u32, u32)>> {
        self.inner.session_overrides().map(|(d, w)| (d, w.to_vec())).collect()
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
    fn weekday_windows(windows: &Bound<'_, PyDict>) -> PyResult<[Vec<(u32, u32)>; 7]> {
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
        Ok(by_day)
    }

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
