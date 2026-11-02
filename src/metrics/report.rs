//! The full report: every figure a desk reads, each one present or explained.
//!
//! Computed on raptorbt's inputs: an equity curve, closed trades, orders and fees.
//! A figure that cannot be computed is absent from `values` (`None` in Python) and named
//! in `unavailable` with a code and a plain reason, never a NaN or a silent zero. Every
//! ratio is a fraction (0.0123 is 1.23%); `costs.bps_per_year` alone is in basis points.
//! Sums are taken with the crate's correctly rounded sum, Python's `math.fsum`.
//!
//! By design:
//! - money is `f64`, because raptorbt's equity curve is. Nothing here is digested.
//! - a trade's `realised` is gross (the ledger convention): raptorbt's
//!   `Trade.pnl` is net of fees, so the caller passes `pnl + fees`.
//! - the fill count and turnover come from orders (`fill_slices`, `filled_qty x
//!   avg_fill_price x multiplier`) rather than ledger fill entries.
//!
//! Not here: rule attribution and the per-trade rows (`metrics::attribution`), and
//! `BacktestResult.full_report`.
//!
//! This module reads nothing of `BacktestMetrics` and changes no existing metric.

use std::collections::{BTreeMap, BTreeSet};

use crate::core::error::{RaptorError, Result};
use crate::math::fsum::fsum;
use crate::metrics::drawdown::DrawdownTracker;
use crate::metrics::streaming::StreamingMetrics;

/// Why a figure is absent. The string forms are [`UNAVAILABLE_CODES`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Unavailable {
    Ruined,
    TooFewSamples,
    ZeroVariance,
    NoDownside,
    NoDrawdown,
    NoTrades,
    NoWins,
    NoLosses,
    NoBenchmark,
    BenchmarkSparse,
    NotRuleBased,
    PooledPosition,
    NoGrossPnl,
}

impl Unavailable {
    pub fn as_str(self) -> &'static str {
        match self {
            Unavailable::Ruined => "RUINED",
            Unavailable::TooFewSamples => "TOO_FEW_SAMPLES",
            Unavailable::ZeroVariance => "ZERO_VARIANCE",
            Unavailable::NoDownside => "NO_DOWNSIDE",
            Unavailable::NoDrawdown => "NO_DRAWDOWN",
            Unavailable::NoTrades => "NO_TRADES",
            Unavailable::NoWins => "NO_WINS",
            Unavailable::NoLosses => "NO_LOSSES",
            Unavailable::NoBenchmark => "NO_BENCHMARK",
            Unavailable::BenchmarkSparse => "BENCHMARK_SPARSE",
            Unavailable::NotRuleBased => "NOT_RULE_BASED",
            Unavailable::PooledPosition => "POOLED_POSITION",
            Unavailable::NoGrossPnl => "NO_GROSS_PNL",
        }
    }
}

/// The unavailable codes, in order.
pub const UNAVAILABLE_CODES: [&str; 13] = [
    "RUINED",
    "TOO_FEW_SAMPLES",
    "ZERO_VARIANCE",
    "NO_DOWNSIDE",
    "NO_DRAWDOWN",
    "NO_TRADES",
    "NO_WINS",
    "NO_LOSSES",
    "NO_BENCHMARK",
    "BENCHMARK_SPARSE",
    "NOT_RULE_BASED",
    "POOLED_POSITION",
    "NO_GROSS_PNL",
];

/// Fee components the `costs` block splits by.
pub const FEE_COMPONENTS: [&str; 8] =
    ["brokerage", "stt", "exchange", "sebi", "stamp", "gst", "ipft", "dp"];

/// A benchmark needs this many aligned returns ...
pub const MIN_BENCHMARK_POINTS: usize = 20;
/// ... and at most this share of curve samples without a benchmark price.
pub const MAX_BENCHMARK_MISSING: f64 = 0.10;
/// Drawdown episodes reported, deepest first.
pub const TOP_DRAWDOWNS: usize = 5;

const YEAR_DAYS: f64 = 365.25;
const DAY_NS: i64 = 86_400_000_000_000;

/// A gross P&L under this share of the starting capital is not divided by: costs of
/// Rs 2,864 on a gross of Rs 0.46 would read as 6,17,941%.
const MIN_GROSS_SHARE: f64 = 0.0001;

/// One report value.
#[derive(Debug, Clone, PartialEq)]
pub enum ReportValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    Map(BTreeMap<String, ReportValue>),
}

/// One closing event: a closed trade, or a settlement at expiry.
#[derive(Debug, Clone, PartialEq)]
pub struct ReportTrade {
    /// Entry instant (ns); `None` when unknown, which leaves it out of holding figures.
    pub opened_at: Option<i64>,
    pub closed_at: i64,
    /// Realised P&L before fees.
    pub realised: f64,
    /// The trade's whole cost, entry and exit.
    pub fees: f64,
    pub tag: String,
    pub settlement: bool,
}

/// One order, for the fill count, turnover and the counters.
#[derive(Debug, Clone, PartialEq)]
pub struct ReportOrder {
    pub filled_qty: f64,
    pub avg_fill_price: Option<f64>,
    /// Contract multiplier, 1 for cash equity.
    pub multiplier: f64,
    pub fill_slices: u32,
    pub status: String,
    pub tag: String,
}

/// Everything the report reads.
#[derive(Debug, Clone)]
pub struct ReportInput<'a> {
    pub ts: &'a [i64],
    pub equity: &'a [f64],
    pub trades: &'a [ReportTrade],
    pub orders: &'a [ReportOrder],
    /// Fees by component. A `total` key is the fees total; without one the total is the
    /// sum of every key. `None`: the total is the trades' fees and no split is known.
    pub fees: Option<&'a BTreeMap<String, f64>>,
    /// Slippage cost over the run, in money (positive is a cost).
    pub slippage: f64,
    /// (ts, gross notional, net notional) samples.
    pub exposure: &'a [(i64, f64, f64)],
    /// (ns in the market, ns observed).
    pub exposure_time: Option<(i64, i64)>,
    /// (ts, price) of a benchmark.
    pub benchmark: Option<&'a [(i64, f64)]>,
    /// The run's first and last day as day numbers (days since 1970-01-01), for
    /// `partial` months; `None` takes them from the curve.
    pub period: Option<(i64, i64)>,
    pub periods_per_year: f64,
    pub risk_free: f64,
    /// Where days and months are cut: ns ahead of UTC (IST is 19,800 s).
    pub tz_offset_ns: i64,
}

/// One drawdown, peak to trough to recovery.
#[derive(Debug, Clone, PartialEq)]
pub struct DrawdownEpisode {
    pub peak_at: i64,
    pub trough_at: i64,
    pub recovered_at: Option<i64>,
    /// (peak - trough) / peak, a positive fraction.
    pub depth: f64,
    pub peak: f64,
    pub trough: f64,
    /// Samples from the peak to the recovery (or the last sample).
    pub sessions: usize,
    /// Calendar days from the peak to the recovery (or the last sample).
    pub days: i64,
    pub recovered: bool,
}

/// One calendar month's or year's return.
#[derive(Debug, Clone, PartialEq)]
pub struct PeriodReturn {
    pub year: i64,
    /// `None` on a yearly row.
    pub month: Option<u32>,
    /// The period's last sample over the previous period's last (the first period: over
    /// the first sample); `None` when that base is not positive.
    pub ret: Option<f64>,
    pub end: f64,
    /// The run does not cover the whole period.
    pub partial: bool,
}

/// The full report.
#[derive(Debug, Clone, PartialEq)]
pub struct FullReport {
    pub values: BTreeMap<String, ReportValue>,
    /// key → (code, reason). Keys of blocks are prefixed: `costs.pct_gross`,
    /// `benchmark.beta`.
    pub unavailable: BTreeMap<String, (Unavailable, String)>,
    /// (ts, drawdown fraction <= 0) at every sample.
    pub drawdown_curve: Vec<(i64, f64)>,
    /// The deepest [`TOP_DRAWDOWNS`] episodes.
    pub drawdowns: Vec<DrawdownEpisode>,
    pub monthly: Vec<PeriodReturn>,
    pub yearly: Vec<PeriodReturn>,
    /// Costs by component and their shares. `None` values are in `unavailable`.
    pub costs: BTreeMap<String, Option<f64>>,
    pub benchmark: Benchmark,
    pub counters: BTreeMap<String, i64>,
}

/// The run against a benchmark. Statistics are `None` when they cannot be computed,
/// with the reason under `benchmark.<key>` in `unavailable`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Benchmark {
    pub beta: Option<f64>,
    pub alpha_annual: Option<f64>,
    pub tracking_error: Option<f64>,
    pub information_ratio: Option<f64>,
    pub correlation: Option<f64>,
    pub total_return: Option<f64>,
    pub excess_return: Option<f64>,
    pub points: usize,
    pub missing: usize,
    /// (ts, the benchmark scaled to the starting equity), when computed.
    pub curve: Option<Vec<(i64, f64)>>,
}

const BENCHMARK_KEYS: [&str; 7] = [
    "beta",
    "alpha_annual",
    "tracking_error",
    "information_ratio",
    "correlation",
    "total_return",
    "excess_return",
];

// --- calendar helpers (integer arithmetic, no zone database) --------------------------------------

/// Day number (days since 1970-01-01) of an instant in a zone `tz` ns ahead of UTC.
pub fn day_of(ts: i64, tz: i64) -> i64 {
    (ts + tz).div_euclid(DAY_NS)
}

/// (year, month, day) of a day number: H. Hinnant's `civil_from_days`.
pub fn civil(n: i64) -> (i64, u32, u32) {
    let z = n + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(m <= 2), m as u32, d as u32)
}

/// Day number of a civil date: [`civil`]'s inverse.
pub fn day_number(y: i64, m: u32, d: u32) -> i64 {
    let y = y - i64::from(m <= 2);
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let m = i64::from(m);
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

// --- the curve ------------------------------------------------------------------------------------

/// Simple returns between consecutive samples, skipping a zero base.
pub fn returns(equity: &[f64]) -> Vec<f64> {
    equity.windows(2).filter(|w| w[0] != 0.0).map(|w| (w[1] - w[0]) / w[0]).collect()
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        fsum(xs.iter().copied()) / xs.len() as f64
    }
}

fn std(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    (fsum(xs.iter().map(|x| (x - m) * (x - m))) / (xs.len() - 1) as f64).sqrt()
}

/// (max drawdown as a positive fraction, its longest run below a peak in samples): the
/// engine's [`DrawdownTracker`].
fn max_drawdown(equity: &[f64]) -> (f64, usize) {
    let mut t = DrawdownTracker::new();
    for &v in equity {
        t.update(v);
    }
    (t.max_drawdown(), t.max_duration())
}

/// Every drawdown episode, deepest first (ties: the earlier peak first).
pub fn drawdown_episodes(ts: &[i64], equity: &[f64], tz: i64) -> Vec<DrawdownEpisode> {
    let episode = |p: usize, t: usize, r: Option<usize>| {
        let (pv, tv) = (equity[p], equity[t]);
        let end = r.unwrap_or(equity.len() - 1);
        DrawdownEpisode {
            peak_at: ts[p],
            trough_at: ts[t],
            recovered_at: r.map(|i| ts[i]),
            depth: if pv > 0.0 { (pv - tv) / pv } else { 0.0 },
            peak: pv,
            trough: tv,
            sessions: end - p,
            days: day_of(ts[end], tz) - day_of(ts[p], tz),
            recovered: r.is_some(),
        }
    };
    let mut eps = Vec::new();
    let (mut peak_i, mut trough_i, mut in_dd) = (0usize, 0usize, false);
    for (i, &v) in equity.iter().enumerate() {
        if v >= equity[peak_i] {
            if in_dd {
                eps.push(episode(peak_i, trough_i, Some(i)));
                in_dd = false;
            }
            peak_i = i;
            trough_i = i;
        } else {
            in_dd = true;
            if v < equity[trough_i] {
                trough_i = i;
            }
        }
    }
    if in_dd {
        eps.push(episode(peak_i, trough_i, None));
    }
    eps.sort_by(|a, b| b.depth.total_cmp(&a.depth).then(a.peak_at.cmp(&b.peak_at)));
    eps
}

/// (ts, drawdown fraction <= 0) at every sample.
pub fn drawdown_curve(ts: &[i64], equity: &[f64]) -> Vec<(i64, f64)> {
    let mut peak = 0.0f64;
    ts.iter()
        .zip(equity)
        .map(|(&t, &v)| {
            peak = peak.max(v);
            (t, if peak > 0.0 { (v - peak) / peak } else { 0.0 })
        })
        .collect()
}

/// Calendar months (`monthly = true`) or years, cut in the zone `tz`.
pub fn period_returns(
    ts: &[i64],
    equity: &[f64],
    tz: i64,
    period: Option<(i64, i64)>,
    monthly: bool,
) -> Vec<PeriodReturn> {
    if ts.is_empty() || equity.is_empty() {
        return Vec::new();
    }
    // (year, month or 0) → last value, in first-seen order
    let mut order: Vec<((i64, u32), f64)> = Vec::new();
    for (&t, &v) in ts.iter().zip(equity) {
        let (y, m, _) = civil(day_of(t, tz));
        let k = (y, if monthly { m } else { 0 });
        // the curve is in time order, so a sample's period is almost always the last one
        let slot = match order.last_mut() {
            Some(last) if last.0 == k => Some(last),
            _ => order.iter_mut().find(|(key, _)| *key == k),
        };
        match slot {
            Some(slot) => slot.1 = v,
            None => order.push((k, v)),
        }
    }
    let (first_day, last_day) =
        period.unwrap_or_else(|| (day_of(ts[0], tz), day_of(ts[ts.len() - 1], tz)));
    let mut prev = equity[0];
    order
        .into_iter()
        .map(|((y, m), v)| {
            let start = day_number(y, m.max(1), 1);
            let end = if m > 0 {
                day_number(y + i64::from(m == 12), m % 12 + 1, 1) - 1
            } else {
                day_number(y, 12, 31)
            };
            let row = PeriodReturn {
                year: y,
                month: (m > 0).then_some(m),
                ret: (prev > 0.0).then(|| (v - prev) / prev),
                end: v,
                partial: first_day > start || last_day < end,
            };
            prev = v;
            row
        })
        .collect()
}

/// Python's `round(x, 4)`: through the shortest decimal, not `x * 1e4`.
fn round4(x: f64) -> f64 {
    format!("{x:.4}").parse().unwrap_or(x)
}

struct Builder {
    values: BTreeMap<String, ReportValue>,
    unavailable: BTreeMap<String, (Unavailable, String)>,
}

impl Builder {
    fn set(&mut self, key: &str, v: ReportValue) {
        self.values.insert(key.to_string(), v);
    }
    fn float(&mut self, key: &str, v: f64) {
        self.set(key, ReportValue::Float(v));
    }
    fn int(&mut self, key: &str, v: i64) {
        self.set(key, ReportValue::Int(v));
    }
    fn na(&mut self, key: &str, code: Unavailable, why: impl Into<String>) {
        self.values.remove(key);
        self.unavailable.insert(key.to_string(), (code, why.into()));
    }
    fn maybe(&mut self, key: &str, v: Option<f64>, code: Unavailable, why: &str) {
        match v {
            Some(x) => self.float(key, x),
            None => self.na(key, code, why),
        }
    }
}

fn meaningful(gross: f64, start: f64) -> bool {
    start > 0.0 && gross > 0.0 && gross >= start * MIN_GROSS_SHARE
}

fn count(n: usize) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}

/// Why every curve figure is absent when the curve has fewer than two samples.
const SHORT_CURVE: &str = "the equity curve has fewer than two samples (an internal run keeps \
                           one a session; curve=\"full\" keeps one an event)";

/// Every figure read off the equity curve: unavailable together when the curve has
/// fewer than two samples.
const CURVE_KEYS: [&str; 16] = [
    "start",
    "end",
    "total_return",
    "calendar_days",
    "annual_return",
    "annual_return_calendar",
    "volatility",
    "sharpe",
    "sortino",
    "max_drawdown",
    "max_drawdown_samples",
    "max_drawdown_sessions",
    "max_drawdown_days",
    "max_drawdown_recovered",
    "calmar",
    "turnover_annual",
];

/// The full report over a run. Fails when the two arrays differ in length or an
/// equity value is not finite. A curve of fewer than two samples (an internal run
/// that kept one session sample) still reports its money and trades; every figure
/// read off the curve is unavailable (`TOO_FEW_SAMPLES`).
pub fn full_report(input: &ReportInput<'_>) -> Result<FullReport> {
    let (ts, equity) = (input.ts, input.equity);
    if ts.len() != equity.len() {
        return Err(RaptorError::length_mismatch(ts.len(), equity.len()));
    }
    if equity.iter().any(|v| !v.is_finite()) {
        return Err(RaptorError::invalid_parameter("an equity sample is not a finite number"));
    }
    // Every curve figure computed below is then replaced by its reason.
    let short = equity.len() < 2;
    let ppy = input.periods_per_year;
    let tz = input.tz_offset_ns;
    let mut b = Builder { values: BTreeMap::new(), unavailable: BTreeMap::new() };

    let (start, end) =
        (equity.first().copied().unwrap_or(0.0), equity.last().copied().unwrap_or(0.0));
    let rs = returns(equity);
    let n = rs.len();
    let total = if start != 0.0 { (end - start) / start } else { 0.0 };
    let ruined = end <= 0.0;
    let years = if ppy != 0.0 { n as f64 / ppy } else { 0.0 };
    let days = if short { 0.0 } else { (ts[ts.len() - 1] - ts[0]) as f64 / 86_400e9 };
    let cal_years = days / YEAR_DAYS;
    b.float("start", start);
    b.float("end", end);
    b.float("total_return", total);
    if ppy.fract() == 0.0 && ppy.abs() < 1e15 {
        b.int("annualisation_basis", ppy as i64);
    } else {
        b.float("annualisation_basis", ppy);
    }
    b.float("risk_free", input.risk_free);
    b.int("samples", count(equity.len()));
    b.float("calendar_days", round4(days));

    let reach_zero = "equity reached zero or below";
    if ruined {
        b.na("annual_return", Unavailable::Ruined, reach_zero);
        b.na("annual_return_calendar", Unavailable::Ruined, reach_zero);
    } else if years <= 0.0 {
        b.na("annual_return", Unavailable::TooFewSamples, "fewer than two equity samples");
        b.na("annual_return_calendar", Unavailable::TooFewSamples, "the run spans no time");
    } else {
        b.float("annual_return", (1.0 + total).powf(1.0 / years) - 1.0);
        b.maybe(
            "annual_return_calendar",
            (cal_years > 0.0).then(|| (1.0 + total).powf(1.0 / cal_years) - 1.0),
            Unavailable::TooFewSamples,
            "the run spans no calendar time",
        );
    }
    let hurdle = if ppy != 0.0 { input.risk_free / ppy } else { 0.0 };
    if n < 2 {
        let why = format!("{n} return(s); at least 2 are needed");
        for k in ["volatility", "sharpe", "sortino"] {
            b.na(k, Unavailable::TooFewSamples, why.clone());
        }
    } else {
        // The engine's streaming metrics define the ratios (sample standard deviation,
        // downside deviation against the per-period risk-free rate), so the report and
        // `BacktestMetrics` read the same Sharpe and Sortino from the same returns.
        let mut m = StreamingMetrics::new().with_target_return(hurdle);
        for &r in &rs {
            m.update(r);
        }
        b.float("volatility", m.std_dev() * ppy.sqrt());
        if m.std_dev() > 0.0 {
            b.float("sharpe", m.sharpe_ratio_with_rf(ppy, input.risk_free));
        } else {
            b.na("sharpe", Unavailable::ZeroVariance, "every period returned the same");
        }
        if m.downside_std_dev() > 0.0 {
            b.float("sortino", m.sortino_ratio(ppy));
        } else {
            b.na("sortino", Unavailable::NoDownside, "no period returned below the risk-free rate");
        }
    }
    let (mdd, mdd_len) = max_drawdown(equity);
    let episodes = drawdown_episodes(ts, equity, tz);
    b.float("max_drawdown", mdd);
    b.int("max_drawdown_samples", count(mdd_len));
    let never_fell = "equity never fell below a previous peak";
    match episodes.first() {
        None => {
            for k in
                ["max_drawdown_sessions", "max_drawdown_days", "max_drawdown_recovered", "calmar"]
            {
                b.na(k, Unavailable::NoDrawdown, never_fell);
            }
        }
        Some(worst) => {
            b.int("max_drawdown_sessions", count(worst.sessions));
            b.int("max_drawdown_days", worst.days);
            b.set("max_drawdown_recovered", ReportValue::Bool(worst.recovered));
            match b.values.get("annual_return") {
                Some(ReportValue::Float(ann)) => {
                    let ann = *ann;
                    b.maybe(
                        "calmar",
                        (mdd > 0.0).then(|| ann / mdd),
                        Unavailable::NoDrawdown,
                        "zero drawdown",
                    );
                }
                _ => b.na("calmar", Unavailable::Ruined, "no annual return to divide"),
            }
        }
    }

    // trades: closed trades and settlements
    let trades = input.trades;
    let fees_total = match input.fees {
        Some(f) => match f.get("total") {
            Some(t) => *t,
            None => fsum(f.values().copied()),
        },
        None => fsum(trades.iter().map(|t| t.fees)),
    };
    let wins: Vec<f64> = trades.iter().map(|t| t.realised).filter(|x| *x > 0.0).collect();
    let losses: Vec<f64> = trades.iter().map(|t| t.realised).filter(|x| *x < 0.0).collect();
    let (gp, gl) = (fsum(wins.iter().copied()), -fsum(losses.iter().copied()));
    let nt = trades.len();
    let fills: i64 = input.orders.iter().map(|o| i64::from(o.fill_slices)).sum();
    let settlements = trades.iter().filter(|t| t.settlement).count();
    b.int("trade_count", count(nt));
    b.int("breakeven", count(nt - wins.len() - losses.len()));
    b.int("fills", fills);
    b.int("settlements", count(settlements));
    b.float("fees", fees_total);
    if nt == 0 {
        for k in ["win_rate", "hit_rate", "expectancy_gross", "expectancy_net"] {
            b.na(k, Unavailable::NoTrades, "no position was closed");
        }
    } else {
        let rate = wins.len() as f64 / nt as f64;
        b.float("win_rate", rate);
        b.float("hit_rate", rate);
        b.float("expectancy_gross", gp / nt as f64 - gl / nt as f64);
        b.float("expectancy_net", (gp - gl - fees_total) / nt as f64);
    }
    if wins.is_empty() {
        b.na("avg_win", Unavailable::NoWins, "no closed trade made money");
    } else {
        b.float("avg_win", gp / wins.len() as f64);
    }
    if losses.is_empty() {
        b.na("avg_loss", Unavailable::NoLosses, "no closed trade lost money");
        b.na("profit_factor", Unavailable::NoLosses, "no losing trade to divide by");
    } else {
        b.float("avg_loss", gl / losses.len() as f64);
        b.float("profit_factor", gp / gl);
    }
    if !wins.is_empty() && !losses.is_empty() {
        b.float("payoff_ratio", (gp / wins.len() as f64) / (gl / losses.len() as f64));
    } else {
        let code = if wins.is_empty() { Unavailable::NoWins } else { Unavailable::NoLosses };
        b.na("payoff_ratio", code, "needs both a winning and a losing trade");
    }
    let held: Vec<(i64, i64)> =
        trades.iter().filter_map(|t| t.opened_at.map(|o| (o, t.closed_at))).collect();
    if held.is_empty() {
        for k in ["holding_minutes_mean", "holding_minutes_median", "holding_sessions_mean"] {
            b.na(k, Unavailable::NoTrades, "no position was closed");
        }
    } else {
        let mut mins: Vec<f64> = held.iter().map(|(o, c)| (c - o) as f64 / 60e9).collect();
        mins.sort_by(f64::total_cmp);
        b.float("holding_minutes_mean", fsum(mins.iter().copied()) / mins.len() as f64);
        let mid = mins.len() / 2;
        let median =
            if mins.len() % 2 == 1 { mins[mid] } else { (mins[mid - 1] + mins[mid]) / 2.0 };
        b.float("holding_minutes_median", median);
        let session_days: Vec<i64> =
            ts.iter().map(|&t| day_of(t, tz)).collect::<BTreeSet<_>>().into_iter().collect();
        let spans = held.iter().map(|&(o, c)| {
            let (a, z) = (day_of(o, tz), day_of(c, tz));
            let inside = session_days.partition_point(|d| *d <= z)
                - session_days.partition_point(|d| *d < a);
            inside.max(1) as f64
        });
        b.float("holding_sessions_mean", fsum(spans) / held.len() as f64);
    }

    // exposure and turnover
    let turnover = fsum(
        input
            .orders
            .iter()
            .filter_map(|o| o.avg_fill_price.map(|px| o.filled_qty.abs() * px * o.multiplier)),
    );
    b.float("turnover", turnover);
    if cal_years > 0.0 {
        b.float("turnover_annual", (turnover / cal_years * 100.0).round_ties_even() / 100.0);
    } else {
        b.na("turnover_annual", Unavailable::TooFewSamples, "the run spans no calendar time");
    }
    match input.exposure_time {
        Some((inside, observed)) if observed > 0 => {
            b.float("exposure_time", inside as f64 / observed as f64)
        }
        _ => b.na("exposure_time", Unavailable::TooFewSamples, "no exposure was sampled"),
    }
    if input.exposure.is_empty() {
        for k in ["gross_exposure_avg", "gross_exposure_max", "net_exposure_avg"] {
            b.na(k, Unavailable::TooFewSamples, "no exposure was sampled");
        }
    } else {
        let e = input.exposure;
        let len = e.len() as f64;
        b.float("gross_exposure_avg", fsum(e.iter().map(|x| x.1)) / len);
        b.float("gross_exposure_max", e.iter().map(|x| x.1).fold(f64::NEG_INFINITY, f64::max));
        b.float("net_exposure_avg", fsum(e.iter().map(|x| x.2)) / len);
    }

    // costs
    let mut costs: BTreeMap<String, Option<f64>> = BTreeMap::new();
    for c in FEE_COMPONENTS {
        let v = input.fees.and_then(|f| f.get(c).copied()).unwrap_or(0.0);
        costs.insert(c.to_string(), Some(v));
    }
    let cost_total = fees_total + input.slippage;
    let net = end - start;
    let gross_pnl = net + cost_total;
    costs.insert("stt_ctt".into(), costs.get("stt").copied().flatten());
    costs.insert("slippage".into(), Some(input.slippage));
    costs.insert("fees_total".into(), Some(fees_total));
    costs.insert("total".into(), Some(cost_total));
    costs.insert("gross_pnl".into(), Some(gross_pnl));
    costs.insert("net_pnl".into(), Some(net));
    let no_gross = "gross P&L is zero, negative, or under 0.01% of the starting capital; \
                    costs cannot be a share of it";
    let shares = [
        ("pct_capital", (start > 0.0).then(|| cost_total / start)),
        ("pct_gross", meaningful(gross_pnl, start).then(|| cost_total / gross_pnl)),
        (
            "bps_per_year",
            (start > 0.0 && cal_years > 0.0).then(|| cost_total / start / cal_years * 10_000.0),
        ),
        ("return_before_costs", (start > 0.0).then(|| gross_pnl / start)),
    ];
    for (k, v) in shares {
        costs.insert(k.into(), v);
        if v.is_none() {
            if k == "pct_gross" && start > 0.0 {
                b.na(&format!("costs.{k}"), Unavailable::NoGrossPnl, no_gross);
            } else {
                b.na(&format!("costs.{k}"), Unavailable::TooFewSamples, "no capital or time");
            }
        }
    }
    b.maybe(
        "fees_pct_of_gross",
        meaningful(gp, start).then(|| fees_total / gp),
        Unavailable::NoGrossPnl,
        "the closed trades' gross profit is zero, or under 0.01% of the starting capital; \
         fees cannot be a share of it",
    );
    b.float("cost_total", cost_total);
    match costs.get("return_before_costs").copied().flatten() {
        Some(v) => b.float("return_before_costs", v),
        None => b.na("return_before_costs", Unavailable::TooFewSamples, "no capital or time"),
    }

    // by tag: realised and fees from the trades, fills from the orders
    let mut by_tag: BTreeMap<String, (Vec<f64>, Vec<f64>, i64)> = BTreeMap::new();
    for t in trades {
        let row = by_tag.entry(t.tag.clone()).or_default();
        row.0.push(t.realised);
        row.1.push(t.fees);
    }
    for o in input.orders.iter().filter(|o| o.fill_slices > 0) {
        by_tag.entry(o.tag.clone()).or_default().2 += i64::from(o.fill_slices);
    }
    let by_tag: BTreeMap<String, ReportValue> = by_tag
        .into_iter()
        .map(|(tag, (realised, fees, fills))| {
            let row = BTreeMap::from([
                ("realised".to_string(), ReportValue::Float(fsum(realised))),
                ("fees".to_string(), ReportValue::Float(fsum(fees))),
                ("fills".to_string(), ReportValue::Int(fills)),
            ]);
            (tag, ReportValue::Map(row))
        })
        .collect();
    b.set("by_tag", ReportValue::Map(by_tag));

    let benchmark = if short {
        for k in BENCHMARK_KEYS {
            b.na(&format!("benchmark.{k}"), Unavailable::TooFewSamples, SHORT_CURVE);
        }
        Benchmark::default()
    } else {
        benchmark_block(ts, equity, input.benchmark, ppy, total, &mut b)
    };
    if short {
        for k in CURVE_KEYS {
            b.na(k, Unavailable::TooFewSamples, SHORT_CURVE);
        }
        for k in ["fees_pct_of_gross", "return_before_costs"] {
            b.na(k, Unavailable::TooFewSamples, SHORT_CURVE);
        }
        // What needs the starting or ending equity, or the run's span.
        for k in [
            "gross_pnl",
            "net_pnl",
            "pct_capital",
            "pct_gross",
            "bps_per_year",
            "return_before_costs",
        ] {
            costs.insert(k.into(), None);
            b.na(&format!("costs.{k}"), Unavailable::TooFewSamples, SHORT_CURVE);
        }
    }

    let mut counters = BTreeMap::new();
    counters.insert("samples".to_string(), count(equity.len()));
    counters.insert("returns".to_string(), count(n));
    counters.insert("trades".to_string(), count(nt));
    counters.insert("settlements".to_string(), count(settlements));
    counters.insert("fills".to_string(), fills);
    counters.insert("orders".to_string(), count(input.orders.len()));
    for o in input.orders {
        *counters.entry(format!("orders_{}", o.status)).or_insert(0) += 1;
    }

    let mut drawdowns = episodes;
    drawdowns.truncate(TOP_DRAWDOWNS);
    let periods = |monthly| {
        if short {
            Vec::new()
        } else {
            period_returns(ts, equity, tz, input.period, monthly)
        }
    };
    Ok(FullReport {
        values: b.values,
        unavailable: b.unavailable,
        drawdown_curve: if short { Vec::new() } else { drawdown_curve(ts, equity) },
        drawdowns: if short { Vec::new() } else { drawdowns },
        monthly: periods(true),
        yearly: periods(false),
        costs,
        benchmark,
        counters,
    })
}

/// The run against a benchmark's prices aligned to the curve (the last price at or
/// before each sample).
fn benchmark_block(
    ts: &[i64],
    equity: &[f64],
    prices: Option<&[(i64, f64)]>,
    ppy: f64,
    total: f64,
    b: &mut Builder,
) -> Benchmark {
    let mut out = Benchmark::default();
    let none = |b: &mut Builder, code: Unavailable, why: &str| {
        for k in BENCHMARK_KEYS {
            b.na(&format!("benchmark.{k}"), code, why);
        }
    };
    let ps = match prices {
        Some(p) if !p.is_empty() => {
            let mut v = p.to_vec();
            v.sort_by(|x, y| x.0.cmp(&y.0).then(x.1.total_cmp(&y.1)));
            v
        }
        _ => {
            none(b, Unavailable::NoBenchmark, "no benchmark was asked for, or it has no prices");
            return out;
        }
    };
    let mut aligned: Vec<Option<f64>> = Vec::with_capacity(ts.len());
    let mut j: Option<usize> = None;
    for &t in ts {
        loop {
            let next = j.map_or(0, |x| x + 1);
            if next < ps.len() && ps[next].0 <= t {
                j = Some(next);
            } else {
                break;
            }
        }
        aligned.push(j.map(|x| ps[x].1));
    }
    let missing = aligned.iter().filter(|a| a.is_none()).count();
    out.missing = missing;
    let mut pairs: Vec<(f64, f64)> = Vec::new();
    for i in 1..ts.len() {
        let (Some(a), Some(bb)) = (aligned[i - 1], aligned[i]) else { continue };
        let (ca, cb) = (equity[i - 1], equity[i]);
        if a == 0.0 || ca == 0.0 {
            continue;
        }
        pairs.push(((bb - a) / a, (cb - ca) / ca));
    }
    out.points = pairs.len();
    if pairs.len() < MIN_BENCHMARK_POINTS
        || missing as f64 > MAX_BENCHMARK_MISSING * ts.len() as f64
    {
        let why = format!(
            "{} aligned returns ({missing} of {} samples without a benchmark price); at least \
             {MIN_BENCHMARK_POINTS} and at most {}% missing are needed",
            pairs.len(),
            ts.len(),
            (MAX_BENCHMARK_MISSING * 100.0) as i64
        );
        none(b, Unavailable::BenchmarkSparse, &why);
        return out;
    }
    let bs: Vec<f64> = pairs.iter().map(|p| p.0).collect();
    let ss: Vec<f64> = pairs.iter().map(|p| p.1).collect();
    let (mb, ms) = (mean(&bs), mean(&ss));
    let dof = (bs.len() - 1) as f64;
    let var_b = fsum(bs.iter().map(|x| (x - mb) * (x - mb))) / dof;
    let cov = fsum(pairs.iter().map(|(x, s)| (x - mb) * (s - ms))) / dof;
    // at least MIN_BENCHMARK_POINTS pairs exist, so both ends have a price
    let first = aligned.iter().flatten().next().copied().unwrap_or(0.0);
    let last = aligned.iter().rev().flatten().next().copied().unwrap_or(0.0);
    let btot = if first != 0.0 { (last - first) / first } else { 0.0 };
    out.total_return = Some(btot);
    out.excess_return = Some(total - btot);
    if first != 0.0 {
        out.curve = Some(
            ts.iter()
                .zip(&aligned)
                .filter_map(|(&t, a)| a.map(|a| (t, equity[0] * a / first)))
                .collect(),
        );
    }
    let te_d: Vec<f64> = pairs.iter().map(|(x, s)| s - x).collect();
    let te = std(&te_d) * ppy.sqrt();
    let (sd_b, sd_s) = (std(&bs), std(&ss));
    if var_b > 0.0 {
        let beta = cov / var_b;
        out.beta = Some(beta);
        out.alpha_annual = Some((ms - beta * mb) * ppy);
    } else {
        for k in ["beta", "alpha_annual"] {
            b.na(&format!("benchmark.{k}"), Unavailable::ZeroVariance, "the benchmark never moved");
        }
    }
    out.tracking_error = Some(te);
    if te > 0.0 {
        out.information_ratio = Some(mean(&te_d) * ppy / te);
    } else {
        b.na(
            "benchmark.information_ratio",
            Unavailable::ZeroVariance,
            "the run tracks the benchmark exactly",
        );
    }
    if sd_b > 0.0 && sd_s > 0.0 {
        out.correlation = Some(cov / (sd_b * sd_s));
    } else {
        b.na(
            "benchmark.correlation",
            Unavailable::ZeroVariance,
            "one of the two series never moved",
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const T0: i64 = 1_767_225_600_000_000_000 + 4 * 3_600_000_000_000; // 2026-01-01 09:30 IST
    const IST: i64 = 19_800_000_000_000;

    fn input<'a>(ts: &'a [i64], eq: &'a [f64], trades: &'a [ReportTrade]) -> ReportInput<'a> {
        ReportInput {
            ts,
            equity: eq,
            trades,
            orders: &[],
            fees: None,
            slippage: 0.0,
            exposure: &[],
            exposure_time: None,
            benchmark: None,
            period: None,
            periods_per_year: 252.0,
            risk_free: 0.0,
            tz_offset_ns: IST,
        }
    }

    fn daily(n: usize) -> Vec<i64> {
        (0..n as i64).map(|i| T0 + i * DAY_NS).collect()
    }

    #[test]
    fn civil_round_trips() {
        assert_eq!(civil(0), (1970, 1, 1));
        assert_eq!(civil(day_number(2026, 2, 28)), (2026, 2, 28));
        assert_eq!(day_number(2024, 3, 1) - day_number(2024, 2, 28), 2);
        for n in -1000..30000 {
            let (y, m, d) = civil(n);
            assert_eq!(day_number(y, m, d), n);
        }
    }

    #[test]
    fn a_flat_curve_explains_every_missing_figure() {
        let ts = daily(3);
        let r = full_report(&input(&ts, &[100.0, 100.0, 100.0], &[])).unwrap();
        for k in ["sharpe", "sortino", "calmar", "win_rate", "profit_factor", "payoff_ratio"] {
            assert!(!r.values.contains_key(k), "{k}");
            assert!(r.unavailable.contains_key(k), "{k}");
        }
        assert_eq!(r.unavailable["sharpe"].0, Unavailable::ZeroVariance);
        assert_eq!(r.unavailable["win_rate"].0, Unavailable::NoTrades);
        assert_eq!(r.unavailable["calmar"].0, Unavailable::NoDrawdown);
        assert_eq!(r.unavailable["benchmark.beta"].0, Unavailable::NoBenchmark);
        assert!(r.unavailable.values().all(|(c, _)| UNAVAILABLE_CODES.contains(&c.as_str())));
    }

    #[test]
    fn a_curve_of_one_sample_reports_its_money_and_explains_every_curve_figure() {
        let trades = [
            ReportTrade {
                opened_at: Some(T0),
                closed_at: T0 + 600_000_000_000,
                realised: 120.0,
                fees: 20.0,
                tag: "a".into(),
                settlement: false,
            },
            ReportTrade {
                opened_at: Some(T0),
                closed_at: T0 + 900_000_000_000,
                realised: -30.0,
                fees: 10.0,
                tag: "b".into(),
                settlement: false,
            },
        ];
        let fees = BTreeMap::from([("brokerage".to_string(), 12.0), ("stt".to_string(), 18.0)]);
        for (ts, eq) in [(vec![T0], vec![100_090.0]), (vec![], vec![])] {
            let mut inp = input(&ts, &eq, &trades);
            inp.fees = Some(&fees);
            let r = full_report(&inp).unwrap();
            // money and trades
            assert_eq!(r.values["fees"], ReportValue::Float(30.0));
            assert_eq!(r.values["trade_count"], ReportValue::Int(2));
            assert_eq!(r.values["win_rate"], ReportValue::Float(0.5));
            assert_eq!(r.values["profit_factor"], ReportValue::Float(4.0));
            assert_eq!(r.costs["brokerage"], Some(12.0));
            assert_eq!(r.costs["fees_total"], Some(30.0));
            assert!(r.values.contains_key("by_tag"));
            assert_eq!(r.counters["trades"], 2);
            // every curve figure is named, none is a number
            for k in
                CURVE_KEYS.iter().chain(&["costs.net_pnl", "costs.pct_gross", "benchmark.beta"])
            {
                assert!(!r.values.contains_key(*k), "{k}");
                assert_eq!(r.unavailable[*k].0, Unavailable::TooFewSamples, "{k}");
            }
            assert_eq!(r.costs["net_pnl"], None);
            assert!(r.monthly.is_empty() && r.yearly.is_empty() && r.drawdowns.is_empty());
            assert!(r.drawdown_curve.is_empty());
            assert!(r.unavailable.values().all(|(c, _)| UNAVAILABLE_CODES.contains(&c.as_str())));
        }
    }

    #[test]
    fn ruin_is_named_not_a_number() {
        let ts = daily(3);
        let r = full_report(&input(&ts, &[100.0, 50.0, -1.0], &[])).unwrap();
        assert_eq!(r.unavailable["annual_return"].0, Unavailable::Ruined);
        assert_eq!(r.unavailable["calmar"].0, Unavailable::Ruined);
    }

    #[test]
    fn drawdown_episodes_depth_duration_and_recovery() {
        let ts = daily(7);
        let eq = [100.0, 90.0, 80.0, 100.0, 120.0, 110.0, 115.0];
        let eps = drawdown_episodes(&ts, &eq, IST);
        assert_eq!(eps.len(), 2);
        assert!((eps[0].depth - 0.2).abs() < 1e-12 && eps[0].recovered);
        assert_eq!((eps[0].sessions, eps[0].days), (3, 3));
        assert!(!eps[1].recovered);
        let r = full_report(&input(&ts, &eq, &[])).unwrap();
        assert_eq!(r.values["max_drawdown_sessions"], ReportValue::Int(3));
        assert_eq!(r.values["max_drawdown_recovered"], ReportValue::Bool(true));
    }

    #[test]
    fn refuses_what_it_cannot_report_on() {
        let ts = daily(2);
        // One sample is reported on (its curve figures explained), not refused.
        assert!(full_report(&input(&ts[..1], &[1.0], &[])).is_ok());
        assert!(full_report(&input(&ts, &[1.0], &[])).is_err());
        assert!(full_report(&input(&ts, &[1.0, f64::NAN], &[])).is_err());
    }
}
