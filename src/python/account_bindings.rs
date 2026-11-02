//! The result of an account group.

use numpy::PyArray1;
use pyo3::prelude::*;

use crate::runtime::groups::GroupOutcome;

/// What an account group ended with: the cash and the account's own curves.
///
/// Each member strategy's `PortfolioResult` holds the strategy's own trades,
/// orders, positions and P&L, and an equity curve attributed to it (see
/// `AccountSpec`). This is the account they shared.
#[pyclass(name = "AccountResult")]
#[derive(Debug, Clone)]
pub struct PyAccountResult {
    /// The account's name in the `accounts` mapping.
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get)]
    pub initial_capital: f64,
    /// Cash at the end, with every member's positions closed. Equals the
    /// initial capital plus the members' trade P&L, which is net of fees.
    #[pyo3(get)]
    pub balance: f64,
    /// Whether a margin call, the drawdown limit, the daily-loss flatten or
    /// `halt_account` halted the account.
    #[pyo3(get)]
    pub halted: bool,
    /// The account's applied-event ordinal when it first halted: events
    /// applied to any member before it.
    #[pyo3(get)]
    pub halted_at: Option<usize>,
    /// Indices, in the call's strategy list, of the strategies on the account,
    /// in the order they step.
    #[pyo3(get, set)]
    pub strategies: Vec<usize>,
    equity_curve: Vec<f64>,
    drawdown_curve: Vec<f64>,
    locked_margin: Vec<f64>,
    timestamps: Vec<i64>,
}

#[pymethods]
impl PyAccountResult {
    /// Account equity (cash plus every member's marks), once per distinct
    /// timestamp; the last value is the final balance.
    fn equity_curve<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.equity_curve.clone())
    }

    /// Peak-to-trough drawdown of the account's equity, in percent.
    fn drawdown_curve<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.drawdown_curve.clone())
    }

    /// Initial margin locked at each equity sample; zero throughout in cash
    /// mode.
    fn locked_margin<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.locked_margin.clone())
    }

    /// Timestamp of each equity sample.
    fn timestamps<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_vec(py, self.timestamps.clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "AccountResult(name={:?}, strategies={}, balance={}, halted={})",
            self.name,
            self.strategies.len(),
            self.balance,
            self.halted
        )
    }
}

pub(crate) fn convert_group(name: String, outcome: GroupOutcome) -> PyAccountResult {
    PyAccountResult {
        name,
        initial_capital: outcome.initial_capital,
        balance: outcome.balance,
        halted: outcome.halted,
        halted_at: outcome.halted_at,
        strategies: outcome.members,
        equity_curve: outcome.equity_curve,
        drawdown_curve: outcome.drawdown_curve,
        locked_margin: outcome.locked_curve,
        timestamps: outcome.timestamps,
    }
}
