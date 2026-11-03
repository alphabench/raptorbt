//! Python bindings for the full report (`raptorbt.full_report`).
//!
//! Inputs are plain sequences, dicts or objects with the named attributes (raptorbt's own
//! `Trade` and `Order` objects qualify); nothing here panics on user input. The figures are
//! `metrics::report`'s and `metrics::attribution`'s; nothing is computed in this file.

use std::collections::BTreeMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::metrics::annualization::{resolve_periods_per_year_with_session, SessionSpec};
use crate::metrics::attribution::{
    attribution as compute_attribution, trade_rows as compute_trade_rows, AttrOrder, AttrRow,
    AttrTrade, RuleMetrics,
};
use crate::metrics::report::{
    day_number, full_report as compute_full_report, Benchmark, DrawdownEpisode, PeriodReturn,
    ReportInput, ReportOrder, ReportTrade, ReportValue,
};

// --- the full report ------------------------------------------------------------------------------

/// A field of a dict item or an attribute of an object; `None` when absent or None.
fn field<'py>(item: &Bound<'py, PyAny>, name: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    let v = match item.cast::<PyDict>() {
        Ok(d) => d.get_item(name)?,
        Err(_) => {
            if item.hasattr(name)? {
                Some(item.getattr(name)?)
            } else {
                None
            }
        }
    };
    Ok(v.filter(|x| !x.is_none()))
}

fn named(name: &str, e: PyErr) -> PyErr {
    PyValueError::new_err(format!("{name}: {e}"))
}

fn required<'py, T: FromPyObjectOwned<'py>>(item: &Bound<'py, PyAny>, name: &str) -> PyResult<T> {
    match field(item, name)? {
        Some(v) => v.extract::<T>().map_err(|e| named(name, e.into())),
        None => Err(PyValueError::new_err(format!("each item needs `{name}`"))),
    }
}

fn optional<'py, T: FromPyObjectOwned<'py>>(
    item: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Option<T>> {
    match field(item, name)? {
        Some(v) => v.extract::<T>().map(Some).map_err(|e| named(name, e.into())),
        None => Ok(None),
    }
}

fn tag_of(item: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(match optional::<String>(item, "tag")? {
        Some(t) => t,
        None => optional::<String>(item, "symbol")?.unwrap_or_default(),
    })
}

/// raptorbt's `Trade.fee_breakdown` names → the report's fee components.
fn component(name: &str) -> &str {
    match name {
        "exchange_txn" => "exchange",
        "sebi_fee" => "sebi",
        "stamp_duty" => "stamp",
        "dp_charge" => "dp",
        other => other,
    }
}

fn add_components(into: &mut BTreeMap<String, f64>, parts: &Bound<'_, PyAny>) -> PyResult<()> {
    let parts =
        parts.cast::<PyDict>().map_err(|_| PyValueError::new_err("fees by component is a dict"))?;
    for (k, v) in parts.iter() {
        let k: String = k.extract()?;
        let v: f64 = v.extract()?;
        *into.entry(component(&k).to_string()).or_insert(0.0) += v;
    }
    Ok(())
}

fn trade_from(item: &Bound<'_, PyAny>) -> PyResult<ReportTrade> {
    let fees = optional::<f64>(item, "fees")?.unwrap_or(0.0);
    let pnl: f64 = required(item, "pnl")?;
    let entry = optional::<i64>(item, "entry_time")?;
    let settlement = optional::<String>(item, "exit_reason")?
        .is_some_and(|r| r.eq_ignore_ascii_case("settlement"));
    Ok(ReportTrade {
        opened_at: entry,
        closed_at: required(item, "exit_time")?,
        // raptorbt's pnl is net of fees; the report classifies on gross
        realised: pnl + fees,
        fees,
        tag: tag_of(item)?,
        settlement,
        open_at_end: optional::<String>(item, "exit_reason")?
            .is_some_and(|r| r.eq_ignore_ascii_case("endofdata")),
    })
}

fn order_from(
    item: &Bound<'_, PyAny>,
    multipliers: Option<&BTreeMap<String, f64>>,
) -> PyResult<ReportOrder> {
    let symbol = optional::<String>(item, "symbol")?.unwrap_or_default();
    let multiplier = match optional::<f64>(item, "multiplier")? {
        Some(m) => m,
        None => multipliers.and_then(|m| m.get(&symbol).copied()).unwrap_or(1.0),
    };
    Ok(ReportOrder {
        filled_qty: optional::<f64>(item, "filled_qty")?.unwrap_or(0.0),
        avg_fill_price: optional(item, "avg_fill_price")?,
        multiplier,
        fill_slices: optional::<u32>(item, "fill_slices")?.unwrap_or(0),
        status: optional::<String>(item, "status")?.unwrap_or_default(),
        tag: tag_of(item)?,
    })
}

/// A `YYYY-MM-DD` string or a `date` (anything with year, month, day) as a day number.
fn period_day(x: &Bound<'_, PyAny>) -> PyResult<i64> {
    if let Ok(s) = x.extract::<String>() {
        let parts: Vec<&str> = s.get(..10).unwrap_or(&s).split('-').collect();
        let bad = || PyValueError::new_err(format!("period bound {s:?} is not YYYY-MM-DD"));
        if parts.len() != 3 {
            return Err(bad());
        }
        let y: i64 = parts[0].parse().map_err(|_| bad())?;
        let m: u32 = parts[1].parse().map_err(|_| bad())?;
        let d: u32 = parts[2].parse().map_err(|_| bad())?;
        return Ok(day_number(y, m, d));
    }
    Ok(day_number(
        x.getattr("year")?.extract()?,
        x.getattr("month")?.extract()?,
        x.getattr("day")?.extract()?,
    ))
}

fn attr_trade_from(item: &Bound<'_, PyAny>) -> PyResult<AttrTrade> {
    let fees = optional::<f64>(item, "fees")?.unwrap_or(0.0);
    let pnl: f64 = required(item, "pnl")?;
    let mut fee_parts = BTreeMap::new();
    if let Some(parts) = field(item, "fee_breakdown")? {
        add_components(&mut fee_parts, &parts)?;
    }
    Ok(AttrTrade {
        id: optional::<i64>(item, "id")?.unwrap_or(0),
        instrument: optional::<String>(item, "symbol")?
            .or(optional::<String>(item, "instrument")?)
            .unwrap_or_default(),
        direction: optional::<i32>(item, "direction")?.unwrap_or(1),
        size: optional::<f64>(item, "size")?.unwrap_or(0.0),
        entry_price: optional::<f64>(item, "entry_price")?.unwrap_or(f64::NAN),
        exit_price: optional::<f64>(item, "exit_price")?.unwrap_or(f64::NAN),
        opened_at: optional::<i64>(item, "entry_time")?,
        closed_at: required(item, "exit_time")?,
        exit_idx: optional::<i64>(item, "exit_idx")?,
        realised: pnl + fees,
        fees,
        entry_fees: optional::<f64>(item, "entry_fees")?.unwrap_or(0.0),
        exit_fees: optional::<f64>(item, "exit_fees")?.unwrap_or(0.0),
        fee_parts,
        tag: tag_of(item)?,
        settlement: optional::<String>(item, "exit_reason")?
            .is_some_and(|r| r.eq_ignore_ascii_case("settlement")),
    })
}

fn attr_order_from(
    item: &Bound<'_, PyAny>,
    multipliers: Option<&BTreeMap<String, f64>>,
) -> PyResult<AttrOrder> {
    let instrument = optional::<String>(item, "symbol")?.unwrap_or_default();
    let multiplier = match optional::<f64>(item, "multiplier")? {
        Some(m) => m,
        None => multipliers.and_then(|m| m.get(&instrument).copied()).unwrap_or(1.0),
    };
    Ok(AttrOrder {
        side: optional::<String>(item, "side")?.unwrap_or_default(),
        source: optional::<String>(item, "source")?,
        filled_qty: optional::<f64>(item, "filled_qty")?.unwrap_or(0.0),
        avg_fill_price: optional(item, "avg_fill_price")?,
        multiplier,
        fill_slices: optional::<u32>(item, "fill_slices")?.unwrap_or(0),
        last_fill_idx: optional::<i64>(item, "last_fill_idx")?,
        instrument,
    })
}

fn rule_metrics_from(obj: &Bound<'_, PyAny>) -> PyResult<RuleMetrics> {
    Ok(RuleMetrics {
        rule_fires: optional::<BTreeMap<String, i64>>(obj, "rule_fires")?.unwrap_or_default(),
        noops_by_source: optional::<BTreeMap<String, BTreeMap<String, i64>>>(
            obj,
            "noops_by_source",
        )?
        .unwrap_or_default(),
    })
}

fn row_to_py<'py>(py: Python<'py>, r: &AttrRow) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("fills", r.fills)?;
    d.set_item("trades", r.trades())?;
    d.set_item("turnover", r.turnover)?;
    d.set_item("realised", r.realised)?;
    d.set_item("fees", r.fees.clone())?;
    // inside raptorbt's fill price; not split out
    d.set_item("slippage", py.None())?;
    d.set_item("wins", r.wins)?;
    d.set_item("losses", r.losses)?;
    d.set_item("gross_profit", r.gross_profit)?;
    d.set_item("gross_loss", r.gross_loss)?;
    Ok(d)
}

fn value_to_py<'py>(py: Python<'py>, v: &ReportValue) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        ReportValue::Bool(b) => b.into_pyobject(py)?.to_owned().into_any(),
        ReportValue::Int(i) => i.into_pyobject(py)?.into_any(),
        ReportValue::Float(f) => f.into_pyobject(py)?.into_any(),
        ReportValue::Map(m) => {
            let d = PyDict::new(py);
            for (k, x) in m {
                d.set_item(k, value_to_py(py, x)?)?;
            }
            d.into_any()
        }
    })
}

fn episode_to_py<'py>(py: Python<'py>, e: &DrawdownEpisode) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("peak_at", e.peak_at)?;
    d.set_item("trough_at", e.trough_at)?;
    d.set_item("recovered_at", e.recovered_at)?;
    d.set_item("depth", e.depth)?;
    d.set_item("peak", e.peak)?;
    d.set_item("trough", e.trough)?;
    d.set_item("sessions", e.sessions)?;
    d.set_item("days", e.days)?;
    d.set_item("recovered", e.recovered)?;
    Ok(d)
}

fn period_to_py<'py>(py: Python<'py>, p: &PeriodReturn) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("year", p.year)?;
    if let Some(m) = p.month {
        d.set_item("month", m)?;
    }
    d.set_item("return", p.ret)?;
    d.set_item("end", p.end)?;
    d.set_item("partial", p.partial)?;
    Ok(d)
}

fn benchmark_to_py<'py>(py: Python<'py>, b: &Benchmark) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("beta", b.beta)?;
    d.set_item("alpha_annual", b.alpha_annual)?;
    d.set_item("tracking_error", b.tracking_error)?;
    d.set_item("information_ratio", b.information_ratio)?;
    d.set_item("correlation", b.correlation)?;
    d.set_item("total_return", b.total_return)?;
    d.set_item("excess_return", b.excess_return)?;
    d.set_item("points", b.points)?;
    d.set_item("missing", b.missing)?;
    if let Some(curve) = &b.curve {
        let rows = PyList::empty(py);
        for (t, e) in curve {
            let row = PyDict::new(py);
            row.set_item("t", t)?;
            row.set_item("equity", e)?;
            rows.append(row)?;
        }
        d.set_item("curve", rows)?;
    }
    Ok(d)
}

/// The full report over an equity curve, its closed trades and orders.
///
/// `periods_per_year=None` infers the basis from the timestamps' spacing and
/// `session_minutes` (`None` or 0 for a market that trades round the clock, 375 for
/// a six-and-a-quarter-hour session), falling back to 252. `tz_offset_ns` places
/// the day boundaries of the monthly and yearly tables (UTC by default). Neither
/// is assumed from any one exchange. `fees=None` takes the
/// fees total from the trades and their `fee_breakdown`s; a dict is fees by component
/// (with an optional `total`). Returns a dict: `report`, `report_unavailable`
/// (`key → "CODE: reason"`), `drawdown_curve`, `drawdowns`, `monthly`, `yearly`, `costs`,
/// `benchmark` and `counters`.
#[pyfunction]
#[pyo3(signature = (
    equity_ts, equity, trades=None, orders=None, fees=None, periods_per_year=None,
    risk_free=0.0, session_minutes=None, tz_offset_ns=0, *,
    slippage=0.0, multipliers=None, exposure=None, exposure_time=None, benchmark=None,
    period=None, attribution=false, rule_metrics=None, unrealised=None
))]
#[allow(clippy::too_many_arguments)]
fn full_report<'py>(
    py: Python<'py>,
    equity_ts: Vec<i64>,
    equity: Vec<f64>,
    trades: Option<&Bound<'py, PyAny>>,
    orders: Option<&Bound<'py, PyAny>>,
    fees: Option<&Bound<'py, PyAny>>,
    periods_per_year: Option<f64>,
    risk_free: f64,
    session_minutes: Option<f64>,
    tz_offset_ns: i64,
    slippage: f64,
    multipliers: Option<BTreeMap<String, f64>>,
    exposure: Option<Vec<(i64, f64, f64)>>,
    exposure_time: Option<(i64, i64)>,
    benchmark: Option<Vec<(i64, f64)>>,
    period: Option<(Bound<'py, PyAny>, Bound<'py, PyAny>)>,
    attribution: bool,
    rule_metrics: Option<&Bound<'py, PyAny>>,
    unrealised: Option<BTreeMap<String, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let mut trade_items: Vec<Bound<'py, PyAny>> = Vec::new();
    if let Some(ts) = trades {
        for item in ts.try_iter()? {
            trade_items.push(item?);
        }
    }
    let mut order_items: Vec<Bound<'py, PyAny>> = Vec::new();
    if let Some(os) = orders {
        for item in os.try_iter()? {
            order_items.push(item?);
        }
    }
    let mut trade_rows = Vec::new();
    let mut breakdown: BTreeMap<String, f64> = BTreeMap::new();
    let mut itemised = 0usize;
    {
        for item in &trade_items {
            trade_rows.push(trade_from(item)?);
            if fees.is_none() {
                if let Some(parts) = field(item, "fee_breakdown")? {
                    add_components(&mut breakdown, &parts)?;
                    itemised += 1;
                }
            }
        }
    }
    let mut order_rows = Vec::new();
    for item in &order_items {
        order_rows.push(order_from(item, multipliers.as_ref())?);
    }
    let fee_map: Option<BTreeMap<String, f64>> = match fees {
        Some(f) => match f.extract::<f64>() {
            Ok(total) => Some(BTreeMap::from([("total".to_string(), total)])),
            Err(_) => {
                let mut m = BTreeMap::new();
                add_components(&mut m, f)?;
                Some(m)
            }
        },
        // the trades' own breakdowns, when every one has one (their `total` is the fees
        // total); otherwise the split is unknown and the total is the trades' fees
        None if itemised > 0 && itemised == trade_rows.len() => Some(breakdown),
        None => None,
    };
    let period = match &period {
        Some((a, b)) => Some((period_day(a)?, period_day(b)?)),
        None => None,
    };
    let session = match session_minutes {
        Some(minutes) if minutes > 0.0 => SessionSpec::Session { minutes },
        _ => SessionSpec::Continuous,
    };
    let ppy = resolve_periods_per_year_with_session(periods_per_year, &equity_ts, session, 252.0);
    let exposure = exposure.unwrap_or_default();
    let input = ReportInput {
        ts: &equity_ts,
        equity: &equity,
        trades: &trade_rows,
        orders: &order_rows,
        fees: fee_map.as_ref(),
        slippage,
        exposure: &exposure,
        exposure_time,
        benchmark: benchmark.as_deref(),
        period,
        periods_per_year: ppy,
        risk_free,
        tz_offset_ns,
    };
    let r = py.detach(|| compute_full_report(&input))?;

    let out = PyDict::new(py);
    let report = PyDict::new(py);
    for (k, v) in &r.values {
        report.set_item(k, value_to_py(py, v)?)?;
    }
    // every unavailable report key is present as None
    let why = PyDict::new(py);
    for (k, (code, text)) in &r.unavailable {
        why.set_item(k, format!("{}: {text}", code.as_str()))?;
        if !k.contains('.') {
            report.set_item(k, py.None())?;
        }
    }
    out.set_item("report", report)?;
    out.set_item("report_unavailable", &why)?;
    let dd = PyList::empty(py);
    for (t, v) in &r.drawdown_curve {
        dd.append(PyList::new(
            py,
            [t.into_pyobject(py)?.into_any(), v.into_pyobject(py)?.into_any()],
        )?)?;
    }
    out.set_item("drawdown_curve", dd)?;
    let eps = PyList::empty(py);
    for e in &r.drawdowns {
        eps.append(episode_to_py(py, e)?)?;
    }
    out.set_item("drawdowns", eps)?;
    for (name, rows) in [("monthly", &r.monthly), ("yearly", &r.yearly)] {
        let l = PyList::empty(py);
        for p in rows {
            l.append(period_to_py(py, p)?)?;
        }
        out.set_item(name, l)?;
    }
    let costs = PyDict::new(py);
    for (k, v) in &r.costs {
        costs.set_item(k, *v)?;
    }
    out.set_item("costs", costs)?;
    out.set_item("benchmark", benchmark_to_py(py, &r.benchmark)?)?;
    let counters = PyDict::new(py);
    for (k, v) in &r.counters {
        counters.set_item(k, v)?;
    }
    out.set_item("counters", counters)?;
    if attribution {
        let attr_trades =
            trade_items.iter().map(attr_trade_from).collect::<PyResult<Vec<AttrTrade>>>()?;
        let attr_orders = order_items
            .iter()
            .map(|o| attr_order_from(o, multipliers.as_ref()))
            .collect::<PyResult<Vec<AttrOrder>>>()?;
        let metrics = rule_metrics.map(rule_metrics_from).transpose()?;
        let a = compute_attribution(&attr_trades, &attr_orders, metrics.as_ref());
        let block = PyDict::new(py);
        let by_inst = PyDict::new(py);
        for (k, row) in &a.by_instrument {
            let d = row_to_py(py, row)?;
            d.set_item("unrealised", unrealised.as_ref().and_then(|u| u.get(k)).copied())?;
            by_inst.set_item(k, d)?;
        }
        match &a.by_rule {
            None => {
                why.set_item(
                    "attribution.by_rule",
                    "NOT_RULE_BASED: a Strategy's orders carry no rule id; see by_instrument",
                )?;
                block.set_item("by_rule", py.None())?;
                block.set_item(
                    "by_rule_reason",
                    "NOT_RULE_BASED: a Strategy's orders carry no rule id",
                )?;
            }
            Some(rules) => {
                why.set_item(
                    "attribution.by_rule.unrealised",
                    "POOLED_POSITION: rules share one position per instrument; open PnL is \
                     reported by instrument only",
                )?;
                let by_rule = PyDict::new(py);
                for (id, r) in rules {
                    let d = row_to_py(py, &r.row)?;
                    d.set_item("fires", r.fires)?;
                    d.set_item("noops", r.noops.clone())?;
                    d.set_item("unrealised", py.None())?;
                    by_rule.set_item(id, d)?;
                }
                block.set_item("by_rule", by_rule)?;
                block.set_item("by_rule_reason", py.None())?;
            }
        }
        block.set_item("by_instrument", by_inst)?;
        out.set_item("attribution", block)?;
        let rows = PyList::empty(py);
        for t in compute_trade_rows(&attr_trades, &a.trade_origins) {
            let d = PyDict::new(py);
            d.set_item("n", t.n)?;
            d.set_item("ts", t.exit_ts)?;
            d.set_item("exit_ts", t.exit_ts)?;
            d.set_item("entry_ts", t.entry_ts)?;
            d.set_item("instrument", t.instrument)?;
            d.set_item("side", t.side)?;
            d.set_item("quantity", t.quantity)?;
            d.set_item("price", t.exit_price)?;
            d.set_item("exit_price", t.exit_price)?;
            d.set_item("entry_price", t.entry_price)?;
            d.set_item("realised", t.realised)?;
            d.set_item("fees", t.fees)?;
            d.set_item("entry_fees", t.entry_fees)?;
            d.set_item("exit_fees", t.exit_fees)?;
            d.set_item("slippage", py.None())?;
            d.set_item("net", t.net)?;
            d.set_item("tag", t.tag)?;
            d.set_item("rule", t.rule)?;
            d.set_item("kind", t.kind)?;
            d.set_item("holding_minutes", t.holding_minutes)?;
            rows.append(d)?;
        }
        out.set_item("trades", rows)?;
    }
    Ok(out)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(full_report, m)?)?;
    Ok(())
}
