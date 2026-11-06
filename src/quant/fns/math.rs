//! Math operators, math transforms, price transforms and statistic functions: the TA-Lib groups
//! that are arithmetic on one bar or a window of bars.
//!
//! Elementwise functions follow the series policy: an infinite result is NaN, never a value
//! (`exp(1000)`, `1 / 0`). Windowed functions read their window from `Hist` and output NaN while
//! it holds a non-finite value.

use crate::quant::eval::ops;
use crate::quant::kernel::{Adapter, Hist, Kernel, Node};
use crate::quant::registry::{
    int_param, regs, Domain, Family, FnDef, InputDef, Invariance, Legacy, Memory, NanPolicy,
    NanRule, OutKind, ParamDef, ParamKind, ParamVal, Reference, Relation,
};
use crate::quant::window::{RollExt, RollSum};

const MAX_PERIOD: i64 = 1 << 24;

/// An infinite value is not a series value.
#[inline(always)]
pub(crate) fn fin(x: f64) -> f64 {
    if x.is_infinite() {
        f64::NAN
    } else {
        x
    }
}

pub(crate) fn float_param(params: &[ParamVal], i: usize) -> Result<f64, String> {
    match params.get(i) {
        Some(ParamVal::Float(v)) => Ok(*v),
        Some(ParamVal::Int(v)) => Ok(*v as f64),
        other => Err(format!("parameter {i} must be a number, got {other:?}")),
    }
}

// ---------------------------------------------------------------- elementwise

/// A stateless function of this bar's inputs. One definition serves batch and streaming.
pub struct MapN<const N: usize>(pub fn([f64; N]) -> f64);

impl<const N: usize> Kernel<N, 1> for MapN<N> {
    fn lookback(&self) -> usize {
        0
    }

    #[inline(always)]
    fn step<H: Hist<N>>(&mut self, x: [f64; N], _h: &H) -> [f64; 1] {
        [(self.0)(x)]
    }
}

macro_rules! elementwise {
    (@make $n:literal, $f:expr) => {{
        fn make(_p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
            let k = MapN::<$n>($f);
            Ok(Box::new(Adapter::new(k, regs::<$n>(ins, "inputs")?, regs::<1>(outs, "outputs")?)))
        }
        make
    }};
    ($name:literal, $talib:literal, $inputs:expr, $n:literal, $doc:expr, $domain:expr,
     $inv:expr, $f:expr) => {
        FnDef {
            name: $name,
            family: Family::Math,
            inputs: $inputs,
            params: &[],
            outputs: &["value"],
            out_kind: OutKind::Series,
            warmup: warm_none,
            memory: Memory::Finite,
            legacy: None,
            talib_name: Some($talib),
            doc: $doc,
            refs: &[],
            rev: 1,
            invariance: $inv,
            causal: true,
            nan_rule: NanRule::Elementwise,
            domain: $domain,
            reference: Reference::Fixture,
            relation: Relation::Exact,
            make: elementwise!(@make $n, $f),
        }
    };
}

const IN_X: &[InputDef] = &[InputDef { name: "x", default_field: Some("close") }];
const IN_AB: &[InputDef] = &[
    InputDef { name: "a", default_field: Some("high") },
    InputDef { name: "b", default_field: Some("low") },
];
const IN_HL: &[InputDef] = &[
    InputDef { name: "high", default_field: Some("high") },
    InputDef { name: "low", default_field: Some("low") },
];
const IN_HLC: &[InputDef] = &[
    InputDef { name: "high", default_field: Some("high") },
    InputDef { name: "low", default_field: Some("low") },
    InputDef { name: "close", default_field: Some("close") },
];
const IN_OHLC: &[InputDef] = &[
    InputDef { name: "open", default_field: Some("open") },
    InputDef { name: "high", default_field: Some("high") },
    InputDef { name: "low", default_field: Some("low") },
    InputDef { name: "close", default_field: Some("close") },
];

const PERIOD_30: &[ParamDef] = &[ParamDef {
    name: "period",
    kind: ParamKind::Int { min: 1, max: MAX_PERIOD },
    default: Some(ParamVal::Int(30)),
}];
const PERIOD_5: &[ParamDef] = &[ParamDef {
    name: "period",
    kind: ParamKind::Int { min: 1, max: MAX_PERIOD },
    default: Some(ParamVal::Int(5)),
}];
const PERIOD_14_FIT: &[ParamDef] = &[ParamDef {
    name: "period",
    kind: ParamKind::Int { min: 2, max: MAX_PERIOD },
    default: Some(ParamVal::Int(14)),
}];
const STDDEV_PARAMS: &[ParamDef] = &[
    ParamDef {
        name: "period",
        kind: ParamKind::Int { min: 1, max: MAX_PERIOD },
        default: Some(ParamVal::Int(5)),
    },
    ParamDef {
        name: "nbdev",
        kind: ParamKind::Float { min: -1e300, max: 1e300 },
        default: Some(ParamVal::Float(1.0)),
    },
];

fn warm_none(_: &[ParamVal]) -> usize {
    0
}

fn warm_period(p: &[ParamVal]) -> usize {
    int_param(p, 0).map_or(0, |v| v.saturating_sub(1))
}

/// A beta window is `period` returns, which take `period + 1` prices.
fn warm_beta(p: &[ParamVal]) -> usize {
    int_param(p, 0).unwrap_or(0)
}

// ---------------------------------------------------------------- rolling sum and extrema

struct Sum {
    rs: RollSum,
    p: usize,
}

impl Kernel<1, 1> for Sum {
    fn lookback(&self) -> usize {
        self.p
    }

    fn step<H: Hist<1>>(&mut self, x: [f64; 1], h: &H) -> [f64; 1] {
        [fin(self.rs.push(x[0], h, 0))]
    }
}

struct Ext {
    e: RollExt,
}

impl Kernel<1, 1> for Ext {
    fn lookback(&self) -> usize {
        0
    }

    fn step<H: Hist<1>>(&mut self, x: [f64; 1], _h: &H) -> [f64; 1] {
        [self.e.push(x[0])]
    }
}

struct MinMax {
    lo: RollExt,
    hi: RollExt,
}

impl Kernel<1, 2> for MinMax {
    fn lookback(&self) -> usize {
        0
    }

    fn step<H: Hist<1>>(&mut self, x: [f64; 1], _h: &H) -> [f64; 2] {
        [self.lo.push(x[0]), self.hi.push(x[0])]
    }
}

/// Absolute bar index of the window's extreme, TA-Lib's tie rule: a value equal to the extreme
/// moves it to the newer bar, but when the extreme leaves the window the window is rescanned
/// and the OLDEST of equal extremes wins. Counting bars from the start of the series makes the
/// answer depend on where the series starts, which is why it matches TA-Lib and is useless
/// live; `bars_since_max` / `bars_since_min` are the relative forms. A rescan is O(period), so
/// a monotone window costs O(period) per bar.
struct ArgTrack {
    p: u64,
    is_max: bool,
    n: u64,
    best: f64,
    at: u64,
    valid: bool,
    last_bad: Option<u64>,
}

impl ArgTrack {
    fn new(p: usize, is_max: bool) -> Self {
        Self { p: p as u64, is_max, n: 0, best: f64::NAN, at: 0, valid: false, last_bad: None }
    }

    fn better(&self, v: f64, than: f64, or_equal: bool) -> bool {
        match (self.is_max, or_equal) {
            (true, false) => v > than,
            (true, true) => v >= than,
            (false, false) => v < than,
            (false, true) => v <= than,
        }
    }

    fn push<H: Hist<1>>(&mut self, x: f64, h: &H) -> f64 {
        let cur = self.n;
        self.n += 1;
        if !x.is_finite() {
            self.last_bad = Some(cur);
        }
        if self.n < self.p || self.last_bad.is_some_and(|b| cur - b < self.p) {
            self.valid = false;
            return f64::NAN;
        }
        if self.valid && self.at + self.p > cur {
            if self.better(x, self.best, true) {
                self.best = x;
                self.at = cur;
            }
        } else {
            let mut best = f64::NAN;
            let mut at = cur;
            for age in (0..self.p as usize).rev() {
                let v = if age == 0 { x } else { h.ago(0, age) };
                if best.is_nan() || self.better(v, best, false) {
                    best = v;
                    at = cur - age as u64;
                }
            }
            self.best = best;
            self.at = at;
            self.valid = true;
        }
        self.at as f64
    }
}

struct ArgOne {
    t: ArgTrack,
    p: usize,
}

impl Kernel<1, 1> for ArgOne {
    fn lookback(&self) -> usize {
        self.p
    }

    fn step<H: Hist<1>>(&mut self, x: [f64; 1], h: &H) -> [f64; 1] {
        [self.t.push(x[0], h)]
    }
}

struct ArgBoth {
    lo: ArgTrack,
    hi: ArgTrack,
    p: usize,
}

impl Kernel<1, 2> for ArgBoth {
    fn lookback(&self) -> usize {
        self.p
    }

    fn step<H: Hist<1>>(&mut self, x: [f64; 1], h: &H) -> [f64; 2] {
        [self.lo.push(x[0], h), self.hi.push(x[0], h)]
    }
}

// ---------------------------------------------------------------- variance

/// Population variance replicating `indicators::streaming::StreamingStdDev` operation for
/// operation: a running sum (add the new value, then subtract the one leaving) gives the mean,
/// then the squared deviations are summed oldest to newest. Differs from the legacy core only
/// on a window of one repeated value, which is exactly 0 here (the legacy mean of a repeated
/// value can be off by an ulp, leaving a variance near 1e-34).
struct PopVar {
    p: usize,
    sum: f64,
    bad: usize,
    n: u64,
    /// Multiplier for the standard deviation; `None` outputs the variance.
    std_scale: Option<f64>,
}

impl Kernel<1, 1> for PopVar {
    fn lookback(&self) -> usize {
        self.p
    }

    fn step<H: Hist<1>>(&mut self, x: [f64; 1], h: &H) -> [f64; 1] {
        let x = x[0];
        self.n += 1;
        if x.is_finite() {
            self.sum += x;
        } else {
            self.bad += 1;
        }
        if self.n > self.p as u64 {
            let old = h.ago(0, self.p);
            if old.is_finite() {
                self.sum -= old;
            } else {
                self.bad -= 1;
            }
        }
        if self.n < self.p as u64 || self.bad > 0 {
            return [f64::NAN];
        }
        let mean = self.sum / self.p as f64;
        let mut acc = 0.0;
        let mut flat = true;
        for age in (0..self.p).rev() {
            let v = if age == 0 { x } else { h.ago(0, age) };
            flat &= v == x;
            acc += (v - mean).powi(2);
        }
        let var = if flat { 0.0 } else { acc / self.p as f64 };
        [match self.std_scale {
            Some(k) => fin(var.sqrt() * k),
            None => fin(var),
        }]
    }
}

// ---------------------------------------------------------------- linear regression

#[derive(Clone, Copy)]
enum LinMode {
    Value,
    Angle,
    Intercept,
    Slope,
    Forecast,
}

/// Least-squares line through the window, x = 0 at the oldest bar. Fitted on deviations from
/// the mean, so a long level offset costs no precision; a window of one repeated value has
/// slope exactly 0.
struct LinReg {
    p: usize,
    mode: LinMode,
}

/// The window mean and slope, or `None` while the window is short or holds a non-finite value.
fn fit_line<H: Hist<1>>(p: usize, x: f64, h: &H) -> Option<(f64, f64)> {
    let mut sum = 0.0;
    let mut flat = true;
    for age in (0..p).rev() {
        let v = if age == 0 { x } else { h.ago(0, age) };
        if !v.is_finite() {
            return None;
        }
        flat &= v == x;
        sum += v;
    }
    if flat {
        return Some((x, 0.0));
    }
    let n = p as f64;
    let mean = sum / n;
    let xbar = (n - 1.0) / 2.0;
    let mut num = 0.0;
    for age in (0..p).rev() {
        let v = if age == 0 { x } else { h.ago(0, age) };
        num += ((p - 1 - age) as f64 - xbar) * (v - mean);
    }
    let sxx = n * (n * n - 1.0) / 12.0;
    Some((mean, num / sxx))
}

impl Kernel<1, 1> for LinReg {
    fn lookback(&self) -> usize {
        self.p
    }

    fn step<H: Hist<1>>(&mut self, x: [f64; 1], h: &H) -> [f64; 1] {
        let Some((mean, slope)) = fit_line(self.p, x[0], h) else {
            return [f64::NAN];
        };
        let n = self.p as f64;
        let xbar = (n - 1.0) / 2.0;
        [fin(match self.mode {
            LinMode::Value => mean + slope * (n - 1.0 - xbar),
            LinMode::Angle => slope.atan() * (180.0 / std::f64::consts::PI),
            LinMode::Intercept => mean - slope * xbar,
            LinMode::Slope => slope,
            LinMode::Forecast => mean + slope * (n - xbar),
        })]
    }
}

// ---------------------------------------------------------------- correl, beta

/// Centred second moments of a window of two series.
pub(crate) struct PairStats {
    pub n: f64,
    pub mx: f64,
    pub my: f64,
    pub sxx: f64,
    pub syy: f64,
    pub sxy: f64,
}

/// The two series' value `age` bars back; with `returns`, their simple return at that bar.
fn pair_at<H: Hist<2>>(h: &H, x: [f64; 2], age: usize, returns: bool) -> Option<[f64; 2]> {
    let val = |i: usize, a: usize| if a == 0 { x[i] } else { h.ago(i, a) };
    let mut out = [0.0; 2];
    for (i, o) in out.iter_mut().enumerate() {
        let cur = val(i, age);
        *o = if returns {
            let prev = val(i, age + 1);
            if !prev.is_finite() || prev == 0.0 {
                return None;
            }
            (cur - prev) / prev
        } else {
            cur
        };
        if !o.is_finite() {
            return None;
        }
    }
    Some(out)
}

/// Population moments of the last `p` values (or `p` returns) of two series, by two passes.
/// `None` while the window is short or holds a non-finite value. A series of one repeated
/// value has exactly zero spread and zero covariance with anything.
pub(crate) fn pair_stats<H: Hist<2>>(
    p: usize,
    x: [f64; 2],
    h: &H,
    returns: bool,
) -> Option<PairStats> {
    let first = pair_at(h, x, p - 1, returns)?;
    let (mut sx, mut sy) = (0.0, 0.0);
    let (mut flat_x, mut flat_y) = (true, true);
    for age in (0..p).rev() {
        let v = pair_at(h, x, age, returns)?;
        flat_x &= v[0] == first[0];
        flat_y &= v[1] == first[1];
        sx += v[0];
        sy += v[1];
    }
    let n = p as f64;
    let (mx, my) = (sx / n, sy / n);
    let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
    for age in (0..p).rev() {
        let v = pair_at(h, x, age, returns)?;
        let (dx, dy) = (v[0] - mx, v[1] - my);
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    if flat_x {
        sxx = 0.0;
    }
    if flat_y {
        syy = 0.0;
    }
    if flat_x || flat_y {
        sxy = 0.0;
    }
    Some(PairStats { n, mx, my, sxx, syy, sxy })
}

/// Spread too small to divide by: under 1e-24 of the squared level, or zero.
pub(crate) fn negligible(s: f64, n: f64, mean: f64) -> bool {
    s.is_nan() || s <= 1e-24 * n * mean * mean
}

struct Correl {
    p: usize,
}

impl Kernel<2, 1> for Correl {
    fn lookback(&self) -> usize {
        self.p
    }

    fn step<H: Hist<2>>(&mut self, x: [f64; 2], h: &H) -> [f64; 1] {
        let Some(s) = pair_stats(self.p, x, h, false) else {
            return [f64::NAN];
        };
        if negligible(s.sxx, s.n, s.mx) || negligible(s.syy, s.n, s.my) {
            return [f64::NAN];
        }
        [fin((s.sxy / (s.sxx.sqrt() * s.syy.sqrt())).clamp(-1.0, 1.0))]
    }
}

/// TA-Lib's beta: the slope of the second input's returns on the first input's returns over
/// `p` returns (`p + 1` prices).
struct Beta {
    p: usize,
}

impl Kernel<2, 1> for Beta {
    fn lookback(&self) -> usize {
        self.p + 1
    }

    fn step<H: Hist<2>>(&mut self, x: [f64; 2], h: &H) -> [f64; 1] {
        let Some(s) = pair_stats(self.p, x, h, true) else {
            return [f64::NAN];
        };
        if negligible(s.sxx, s.n, s.mx) {
            return [f64::NAN];
        }
        [fin(s.sxy / s.sxx)]
    }
}

// ---------------------------------------------------------------- makers

fn period_of(p: &[ParamVal]) -> Result<usize, String> {
    let v = int_param(p, 0)?;
    if v == 0 {
        return Err("period must be at least 1".into());
    }
    Ok(v)
}

fn make_sum(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let p = period_of(p)?;
    let k = Sum { rs: RollSum::new(p), p };
    Ok(Box::new(Adapter::new(k, regs::<1>(ins, "sum inputs")?, regs::<1>(outs, "sum outputs")?)))
}

fn make_max(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = Ext { e: RollExt::new(period_of(p)?, true) };
    Ok(Box::new(Adapter::new(k, regs::<1>(ins, "max inputs")?, regs::<1>(outs, "max outputs")?)))
}

fn make_min(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = Ext { e: RollExt::new(period_of(p)?, false) };
    Ok(Box::new(Adapter::new(k, regs::<1>(ins, "min inputs")?, regs::<1>(outs, "min outputs")?)))
}

fn make_minmax(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let p = period_of(p)?;
    let k = MinMax { lo: RollExt::new(p, false), hi: RollExt::new(p, true) };
    Ok(Box::new(Adapter::new(
        k,
        regs::<1>(ins, "minmax inputs")?,
        regs::<2>(outs, "minmax outputs")?,
    )))
}

fn make_maxindex(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let p = period_of(p)?;
    let k = ArgOne { t: ArgTrack::new(p, true), p };
    Ok(Box::new(Adapter::new(
        k,
        regs::<1>(ins, "maxindex inputs")?,
        regs::<1>(outs, "maxindex outputs")?,
    )))
}

fn make_minindex(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let p = period_of(p)?;
    let k = ArgOne { t: ArgTrack::new(p, false), p };
    Ok(Box::new(Adapter::new(
        k,
        regs::<1>(ins, "minindex inputs")?,
        regs::<1>(outs, "minindex outputs")?,
    )))
}

fn make_minmaxindex(
    p: &[ParamVal],
    ins: &[usize],
    outs: &[usize],
) -> Result<Box<dyn Node>, String> {
    let p = period_of(p)?;
    let k = ArgBoth { lo: ArgTrack::new(p, false), hi: ArgTrack::new(p, true), p };
    Ok(Box::new(Adapter::new(
        k,
        regs::<1>(ins, "minmaxindex inputs")?,
        regs::<2>(outs, "minmaxindex outputs")?,
    )))
}

fn make_var(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let p = period_of(p)?;
    let k = PopVar { p, sum: 0.0, bad: 0, n: 0, std_scale: None };
    Ok(Box::new(Adapter::new(k, regs::<1>(ins, "var inputs")?, regs::<1>(outs, "var outputs")?)))
}

fn make_stddev(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k =
        PopVar { p: period_of(p)?, sum: 0.0, bad: 0, n: 0, std_scale: Some(float_param(p, 1)?) };
    Ok(Box::new(Adapter::new(
        k,
        regs::<1>(ins, "stddev inputs")?,
        regs::<1>(outs, "stddev outputs")?,
    )))
}

macro_rules! make_linreg {
    ($fname:ident, $mode:expr, $label:literal) => {
        fn $fname(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
            let k = LinReg { p: period_of(p)?, mode: $mode };
            Ok(Box::new(Adapter::new(
                k,
                regs::<1>(ins, concat!($label, " inputs"))?,
                regs::<1>(outs, concat!($label, " outputs"))?,
            )))
        }
    };
}
make_linreg!(make_linearreg, LinMode::Value, "linearreg");
make_linreg!(make_linearreg_angle, LinMode::Angle, "linearreg_angle");
make_linreg!(make_linearreg_intercept, LinMode::Intercept, "linearreg_intercept");
make_linreg!(make_linearreg_slope, LinMode::Slope, "linearreg_slope");
make_linreg!(make_tsf, LinMode::Forecast, "tsf");

fn make_correl(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = Correl { p: period_of(p)? };
    Ok(Box::new(Adapter::new(
        k,
        regs::<2>(ins, "correl inputs")?,
        regs::<1>(outs, "correl outputs")?,
    )))
}

fn make_beta(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = Beta { p: period_of(p)? };
    Ok(Box::new(Adapter::new(k, regs::<2>(ins, "beta inputs")?, regs::<1>(outs, "beta outputs")?)))
}

// ---------------------------------------------------------------- the table

const SCALE_SHIFT_EQ: &[Invariance] = &[Invariance::ScaleEquivariant, Invariance::ShiftEquivariant];
const SCALE_SHIFT_INV: &[Invariance] = &[Invariance::ScaleInvariant, Invariance::ShiftInvariant];

/// A windowed function of the one input `x` (TA-Lib's `close`).
macro_rules! windowed {
    ($name:literal, $talib:literal, $params:expr, $doc:expr, $inv:expr, $make:expr) => {
        windowed!($name, $talib, $params, $doc, $inv, $make, Relation::Exact, None, &["value"])
    };
    ($name:literal, $talib:literal, $params:expr, $doc:expr, $inv:expr, $make:expr,
     $relation:expr, $legacy:expr, $outputs:expr) => {
        FnDef {
            name: $name,
            family: Family::Math,
            inputs: IN_X,
            params: $params,
            outputs: $outputs,
            out_kind: OutKind::Series,
            warmup: warm_period,
            memory: Memory::Finite,
            legacy: $legacy,
            talib_name: Some($talib),
            doc: $doc,
            refs: &[],
            rev: 1,
            invariance: $inv,
            causal: true,
            nan_rule: NanRule::Windowed,
            domain: Domain::Any,
            reference: Reference::Fixture,
            relation: $relation,
            make: $make,
        }
    };
}

pub const DEFS: &[FnDef] = &[
    // Math operators
    elementwise!("add", "ADD", IN_AB, 2, "a + b.", Domain::Any, &[Invariance::ScaleEquivariant],
        |x| fin(ops::add(x[0], x[1]))),
    elementwise!("sub", "SUB", IN_AB, 2, "a - b.", Domain::Any,
        &[Invariance::ScaleEquivariant, Invariance::ShiftInvariant],
        |x| fin(ops::sub(x[0], x[1]))),
    elementwise!("mult", "MULT", IN_AB, 2, "a * b.", Domain::Any, &[],
        |x| fin(ops::mul(x[0], x[1]))),
    elementwise!("div", "DIV", IN_AB, 2, "a / b; a zero divisor gives NaN, never infinity.",
        Domain::Any, &[Invariance::ScaleInvariant], |x| ops::div(x[0], x[1])),
    windowed!("max", "MAX", PERIOD_30,
        "Highest value of the last `period` values.", SCALE_SHIFT_EQ, make_max),
    windowed!("min", "MIN", PERIOD_30,
        "Lowest value of the last `period` values.", SCALE_SHIFT_EQ, make_min),
    windowed!("sum", "SUM", PERIOD_30,
        "Sum of the last `period` values (compensated).", &[Invariance::ScaleEquivariant],
        make_sum),
    windowed!("minmax", "MINMAX", PERIOD_30,
        "Lowest and highest value of the last `period` values.", SCALE_SHIFT_EQ, make_minmax,
        Relation::Exact, None, &["min", "max"]),
    windowed!("maxindex", "MAXINDEX", PERIOD_30,
        concat!("Absolute bar index of the highest value of the last `period` values. ",
            "The index counts from the first bar of the series it is run on, so it changes with ",
            "where the series starts: correct for TA-Lib parity, useless live. Use ",
            "bars_since_max for the causal relative form. Ties follow TA-Lib."),
        SCALE_SHIFT_INV, make_maxindex),
    windowed!("minindex", "MININDEX", PERIOD_30,
        concat!("Absolute bar index of the lowest value of the last `period` values. ",
            "The index counts from the first bar of the series it is run on, so it changes with ",
            "where the series starts: correct for TA-Lib parity, useless live. Use ",
            "bars_since_min for the causal relative form. Ties follow TA-Lib."),
        SCALE_SHIFT_INV, make_minindex),
    windowed!("minmaxindex", "MINMAXINDEX", PERIOD_30,
        concat!("Absolute bar indexes of the lowest and highest values of the last `period` ",
            "values. The indexes count from the first bar of the series they are run on, so ",
            "they change with where the series starts: correct for TA-Lib parity, useless ",
            "live. Use bars_since_min / bars_since_max for the causal relative form. Ties ",
            "follow TA-Lib."),
        SCALE_SHIFT_INV, make_minmaxindex, Relation::Exact, None, &["minidx", "maxidx"]),
    // Math transforms
    elementwise!("acos", "ACOS", IN_X, 1, "Arc cosine; outside [-1, 1] is NaN.", Domain::Any, &[],
        |x| fin(x[0].acos())),
    elementwise!("asin", "ASIN", IN_X, 1, "Arc sine; outside [-1, 1] is NaN.", Domain::Any, &[],
        |x| fin(x[0].asin())),
    elementwise!("atan", "ATAN", IN_X, 1, "Arc tangent, radians.", Domain::Any, &[],
        |x| fin(x[0].atan())),
    elementwise!("ceil", "CEIL", IN_X, 1, "Smallest integer not below x.", Domain::Any, &[],
        |x| fin(x[0].ceil())),
    elementwise!("cos", "COS", IN_X, 1, "Cosine, radians.", Domain::Any, &[],
        |x| fin(x[0].cos())),
    elementwise!("cosh", "COSH", IN_X, 1, "Hyperbolic cosine; overflow is NaN.", Domain::Any, &[],
        |x| fin(x[0].cosh())),
    elementwise!("exp", "EXP", IN_X, 1, "e to the power x; overflow is NaN, never infinity.",
        Domain::Any, &[], |x| fin(x[0].exp())),
    elementwise!("floor", "FLOOR", IN_X, 1, "Largest integer not above x.", Domain::Any, &[],
        |x| fin(x[0].floor())),
    elementwise!("ln", "LN", IN_X, 1, "Natural logarithm; zero and negative are NaN.",
        Domain::Positive, &[], |x| ops::ln(x[0])),
    elementwise!("log10", "LOG10", IN_X, 1, "Base-10 logarithm; zero and negative are NaN.",
        Domain::Positive, &[], |x| fin(x[0].log10())),
    elementwise!("sin", "SIN", IN_X, 1, "Sine, radians.", Domain::Any, &[],
        |x| fin(x[0].sin())),
    elementwise!("sinh", "SINH", IN_X, 1, "Hyperbolic sine; overflow is NaN.", Domain::Any, &[],
        |x| fin(x[0].sinh())),
    elementwise!("sqrt", "SQRT", IN_X, 1, "Square root; negative is NaN.", Domain::Any, &[],
        |x| fin(x[0].sqrt())),
    elementwise!("tan", "TAN", IN_X, 1, "Tangent, radians.", Domain::Any, &[],
        |x| fin(x[0].tan())),
    elementwise!("tanh", "TANH", IN_X, 1, "Hyperbolic tangent.", Domain::Any, &[],
        |x| fin(x[0].tanh())),
    // Price transforms
    elementwise!("avgprice", "AVGPRICE", IN_OHLC, 4, "(open + high + low + close) / 4.",
        Domain::Any, SCALE_SHIFT_EQ, |x| fin((x[0] + x[1] + x[2] + x[3]) / 4.0)),
    elementwise!("medprice", "MEDPRICE", IN_HL, 2, "(high + low) / 2.", Domain::Any,
        SCALE_SHIFT_EQ, |x| fin((x[0] + x[1]) / 2.0)),
    elementwise!("typprice", "TYPPRICE", IN_HLC, 3, "(high + low + close) / 3.", Domain::Any,
        SCALE_SHIFT_EQ, |x| fin((x[0] + x[1] + x[2]) / 3.0)),
    elementwise!("wclprice", "WCLPRICE", IN_HLC, 3, "(high + low + 2 * close) / 4.", Domain::Any,
        SCALE_SHIFT_EQ, |x| fin((x[0] + x[1] + 2.0 * x[2]) / 4.0)),
    // Statistic functions
    FnDef {
        name: "beta",
        family: Family::Math,
        inputs: IN_HL,
        params: PERIOD_5,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warm_beta,
        memory: Memory::Finite,
        legacy: None,
        talib_name: Some("BETA"),
        doc: "Beta of the second input on the first: covariance of their simple returns over the last `period` returns divided by the variance of the first's. Defined on returns, not levels; first valid at bar `period`. A first input with no movement gives NaN (TA-Lib gives 0).",
        refs: &[],
        rev: 1,
        invariance: &[Invariance::ScaleInvariant],
        causal: true,
        nan_rule: NanRule::Windowed,
        domain: Domain::Any,
        reference: Reference::Fixture,
        // Measured against the committed fixtures with an exact two-pass computation: largest
        // relative difference 7.4e-11.
        relation: Relation::Exact,
        make: make_beta,
    },
    FnDef {
        name: "correl",
        family: Family::Math,
        inputs: IN_HL,
        params: PERIOD_30,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warm_period,
        memory: Memory::Finite,
        legacy: None,
        talib_name: Some("CORREL"),
        doc: "Pearson correlation of the two inputs over the last `period` values. A flat window gives NaN (TA-Lib gives 0).",
        refs: &[],
        rev: 1,
        invariance: SCALE_SHIFT_INV,
        causal: true,
        nan_rule: NanRule::Windowed,
        domain: Domain::Any,
        reference: Reference::Fixture,
        // TA-Lib accumulates raw sums of squares; on the tight_range fixture its own rounding
        // error reaches 1e-6 relative, so the difference is pinned, not asserted exact.
        relation: Relation::DefinedDifference,
        make: make_correl,
    },
    windowed!("linearreg", "LINEARREG", PERIOD_14_FIT,
        "Value at the newest bar of the least-squares line through the last `period` values.",
        SCALE_SHIFT_EQ, make_linearreg),
    windowed!("linearreg_angle", "LINEARREG_ANGLE", PERIOD_14_FIT,
        "Angle in degrees of the least-squares line through the last `period` values: atan(slope) * 180 / pi, with the slope per bar in the input's own units.",
        &[], make_linearreg_angle, Relation::DefinedDifference, None, &["value"]),
    windowed!("linearreg_intercept", "LINEARREG_INTERCEPT", PERIOD_14_FIT,
        "Intercept (value at the oldest bar of the window) of the least-squares line through the last `period` values.",
        SCALE_SHIFT_EQ, make_linearreg_intercept),
    windowed!("linearreg_slope", "LINEARREG_SLOPE", PERIOD_14_FIT,
        "Slope per bar of the least-squares line through the last `period` values. A flat window has slope exactly 0.",
        &[Invariance::ScaleEquivariant, Invariance::ShiftInvariant], make_linearreg_slope,
        Relation::DefinedDifference, None, &["value"]),
    windowed!("stddev", "STDDEV", STDDEV_PARAMS,
        "Population standard deviation of the last `period` values (divisor `period`), times `nbdev`. A flat window is exactly 0.",
        &[Invariance::ShiftInvariant], make_stddev, Relation::DefinedDifference,
        Some(Legacy {
            nan: NanPolicy::Recover,
            pinned_to: "indicators::streaming::StreamingStdDev",
            tolerance: 1e-12,
        }), &["value"]),
    windowed!("tsf", "TSF", PERIOD_14_FIT,
        "Time series forecast: the least-squares line through the last `period` values, one bar past the newest.",
        SCALE_SHIFT_EQ, make_tsf),
    windowed!("var", "VAR", PERIOD_5,
        "Population variance of the last `period` values (divisor `period`). A flat window is exactly 0.",
        &[Invariance::ShiftInvariant], make_var, Relation::DefinedDifference, None, &["value"]),
];

#[cfg(test)]
pub(crate) mod testkit {
    use super::*;
    use crate::quant::kernel::{canon_bits, Regs};
    use crate::quant::registry::{lookup, FnDef};
    use proptest::prelude::*;
    use proptest::test_runner::{Config, RngAlgorithm, RngSeed, TestRunner};

    fn node(name: &str, params: &[ParamVal]) -> (Box<dyn Node>, usize, usize) {
        let def = lookup(name).unwrap();
        let (i, o) = (def.inputs.len(), def.outputs.len());
        let ins: Vec<usize> = (0..i).collect();
        let outs: Vec<usize> = (i..i + o).collect();
        ((def.make)(params, &ins, &outs).unwrap(), i, o)
    }

    /// One column per output.
    pub(crate) fn batch(name: &str, params: &[ParamVal], cols: &[&[f64]]) -> Vec<Vec<f64>> {
        let (mut n, _, o) = node(name, params);
        let len = cols[0].len();
        let mut out = vec![vec![0.0; len]; o];
        {
            let mut refs: Vec<&mut [f64]> = out.iter_mut().map(|v| &mut v[..]).collect();
            n.run_batch(cols, &mut refs);
        }
        out
    }

    pub(crate) fn stream(name: &str, params: &[ParamVal], cols: &[&[f64]]) -> Vec<Vec<f64>> {
        let (mut n, i, o) = node(name, params);
        let len = cols[0].len();
        let mut regs = Regs::new(i + o);
        let mut out = vec![vec![0.0; len]; o];
        for t in 0..len {
            for (j, c) in cols.iter().enumerate() {
                regs.set(j, c[t]);
            }
            n.step(&mut regs);
            for (k, col) in out.iter_mut().enumerate() {
                col[t] = regs.get(i + k);
            }
        }
        out
    }

    pub(crate) fn bits(v: &[Vec<f64>]) -> Vec<Vec<u64>> {
        v.iter().map(|c| c.iter().map(|&x| canon_bits(x)).collect()).collect()
    }

    /// A valid parameter set: a small value inside each range.
    pub(crate) fn sample_params(def: &FnDef, int_pick: i64) -> Vec<ParamVal> {
        def.params
            .iter()
            .map(|p| match p.kind {
                ParamKind::Int { min, max } => ParamVal::Int(int_pick.max(min).min(max)),
                ParamKind::Float { min, max } => ParamVal::Float(0.5f64.max(min).min(max)),
                ParamKind::Enum(v) => ParamVal::Enum(v[0]),
            })
            .collect()
    }

    /// Batch and streaming agree bit for bit with NaN and infinities in the data, no output is
    /// infinite, and the outputs before a cut do not move when everything after it is replaced.
    pub(crate) fn check_table(defs: &'static [FnDef], seed: u64) {
        let value = prop_oneof![8 => -50.0f64..50.0, 1 => Just(f64::NAN), 1 => Just(f64::INFINITY)];
        let strategy = (
            prop::collection::vec(value.clone(), 0..120),
            prop::collection::vec(value.clone(), 0..120),
            prop::collection::vec(value, 0..120),
            1i64..9,
            0usize..120,
        );
        let mut runner = TestRunner::new(Config {
            cases: 60,
            failure_persistence: None,
            rng_algorithm: RngAlgorithm::ChaCha,
            rng_seed: RngSeed::Fixed(seed),
            ..Config::default()
        });
        runner
            .run(&strategy, |(a, b, c, pick, cut)| {
                let n = a.len().min(b.len()).min(c.len());
                let cut = cut.min(n);
                for def in defs {
                    let params = sample_params(def, pick);
                    let all = [&a[..n], &b[..n], &c[..n], &a[..n]];
                    let cols = &all[..def.inputs.len()];
                    let got = batch(def.name, &params, cols);
                    prop_assert_eq!(
                        bits(&got),
                        bits(&stream(def.name, &params, cols)),
                        "{} {:?}",
                        def.name,
                        params
                    );
                    for out in &got {
                        prop_assert!(out.iter().all(|v| !v.is_infinite()), "{} gave inf", def.name);
                    }
                    let poisoned: Vec<Vec<f64>> = [&a, &b, &c, &a]
                        .iter()
                        .map(|s| {
                            (0..n)
                                .map(|i| if i < cut { s[i] } else { 1e9 + (i * i) as f64 })
                                .collect()
                        })
                        .collect();
                    let pall =
                        [&poisoned[0][..], &poisoned[1][..], &poisoned[2][..], &poisoned[3][..]];
                    let moved = batch(def.name, &params, &pall[..def.inputs.len()]);
                    for (x, y) in got.iter().zip(&moved) {
                        prop_assert_eq!(
                            bits(&[x[..cut].to_vec()]),
                            bits(&[y[..cut].to_vec()]),
                            "{} prefix",
                            def.name
                        );
                    }
                }
                Ok(())
            })
            .unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::testkit::*;
    use super::*;
    use crate::quant::registry::lookup;

    const NAN: f64 = f64::NAN;

    fn one(name: &str, params: &[ParamVal], cols: &[&[f64]]) -> Vec<f64> {
        batch(name, params, cols).remove(0)
    }

    fn p(period: i64) -> Vec<ParamVal> {
        vec![ParamVal::Int(period)]
    }

    fn same(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g.is_nan() && w.is_nan()) || (g - w).abs() <= 1e-12 * w.abs().max(1.0),
                "bar {i}: got {g}, want {w}"
            );
        }
    }

    #[test]
    fn every_math_function_agrees_between_batch_and_streaming() {
        check_table(DEFS, 31);
    }

    #[test]
    fn registry_rows_are_complete() {
        for d in DEFS {
            assert!(d.talib_name.is_some(), "{}", d.name);
            assert_eq!(d.family, Family::Math);
        }
        assert_eq!(DEFS.len(), 39);
        assert!(lookup("maxindex").unwrap().doc.contains("useless live"));
    }

    #[test]
    fn operators_hand_values() {
        let (a, b) = ([6.0, 1.0, 2.0], [3.0, 0.0, 8.0]);
        same(&one("add", &[], &[&a, &b]), &[9.0, 1.0, 10.0]);
        same(&one("sub", &[], &[&a, &b]), &[3.0, 1.0, -6.0]);
        same(&one("mult", &[], &[&a, &b]), &[18.0, 0.0, 16.0]);
        // A zero divisor is NaN, not infinity.
        same(&one("div", &[], &[&a, &b]), &[2.0, NAN, 0.25]);
    }

    #[test]
    fn transforms_hand_values() {
        let x = [1.0, 0.0, -4.0, 2.0, 1000.0];
        same(&one("ln", &[], &[&x]), &[0.0, NAN, NAN, 2f64.ln(), 1000f64.ln()]);
        same(&one("sqrt", &[], &[&x]), &[1.0, 0.0, NAN, 2f64.sqrt(), 1000f64.sqrt()]);
        same(&one("acos", &[], &[&x]), &[0.0, std::f64::consts::FRAC_PI_2, NAN, NAN, NAN]);
        same(&one("exp", &[], &[&[0.0, 1000.0]]), &[1.0, NAN]);
        same(&one("ceil", &[], &[&[1.2, -1.2]]), &[2.0, -1.0]);
        same(&one("floor", &[], &[&[1.2, -1.2]]), &[1.0, -2.0]);
        same(&one("log10", &[], &[&[1000.0, 0.0, -1.0]]), &[3.0, NAN, NAN]);
        same(&one("tanh", &[], &[&[0.0]]), &[0.0]);
    }

    #[test]
    fn price_transforms_hand_values() {
        let (o, h, l, c) = ([1.0], [4.0], [2.0], [3.0]);
        // (1 + 4 + 2 + 3) / 4, (4 + 2) / 2, (4 + 2 + 3) / 3, (4 + 2 + 2 * 3) / 4
        same(&one("avgprice", &[], &[&o, &h, &l, &c]), &[2.5]);
        same(&one("medprice", &[], &[&h, &l]), &[3.0]);
        same(&one("typprice", &[], &[&h, &l, &c]), &[3.0]);
        same(&one("wclprice", &[], &[&h, &l, &c]), &[3.0]);
    }

    #[test]
    fn window_max_min_sum_hand_values() {
        let x = [1.0, 3.0, 2.0, 5.0, 4.0];
        same(&one("max", &p(3), &[&x]), &[NAN, NAN, 3.0, 5.0, 5.0]);
        same(&one("min", &p(3), &[&x]), &[NAN, NAN, 1.0, 2.0, 2.0]);
        same(&one("sum", &p(3), &[&x]), &[NAN, NAN, 6.0, 10.0, 11.0]);
        let mm = batch("minmax", &p(3), &[&x]);
        same(&mm[0], &[NAN, NAN, 1.0, 2.0, 2.0]);
        same(&mm[1], &[NAN, NAN, 3.0, 5.0, 5.0]);
    }

    #[test]
    fn index_functions_hand_values() {
        // Windows [1,3,2] [3,2,5] [2,5,4]: the highest sits at bars 1, 3, 3; the lowest at 0, 2, 2.
        let x = [1.0, 3.0, 2.0, 5.0, 4.0];
        same(&one("maxindex", &p(3), &[&x]), &[NAN, NAN, 1.0, 3.0, 3.0]);
        same(&one("minindex", &p(3), &[&x]), &[NAN, NAN, 0.0, 2.0, 2.0]);
        let both = batch("minmaxindex", &p(3), &[&x]);
        same(&both[0], &[NAN, NAN, 0.0, 2.0, 2.0]);
        same(&both[1], &[NAN, NAN, 1.0, 3.0, 3.0]);
    }

    #[test]
    fn index_ties_follow_ta_lib() {
        // Expected values were measured on TA-Lib 0.6.4 (MAXINDEX / MININDEX, timeperiod 4) for
        // this input; they exercise both tie rules (a newer equal value takes over, but a
        // rescan after the extreme leaves keeps the oldest of equals).
        let x = [1.0, 5.0, 5.0, 3.0, 3.0, 3.0, 2.0, 1.0, 1.0, 0.0, 4.0, 4.0];
        let hi = [NAN, NAN, NAN, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 11.0];
        let lo = [NAN, NAN, NAN, 0.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.0, 9.0];
        same(&one("maxindex", &p(4), &[&x]), &hi);
        same(&one("minindex", &p(4), &[&x]), &lo);
        let both = batch("minmaxindex", &p(4), &[&x]);
        same(&both[0], &lo);
        same(&both[1], &hi);
    }

    #[test]
    fn index_is_absolute_so_it_moves_with_the_series_start() {
        // The window [5, 3, 3, 3] has its highest value at its first bar: index 1 when two
        // bars precede it, index 0 when the series starts there.
        let x = [1.0, 5.0, 5.0, 3.0, 3.0, 3.0];
        assert_eq!(one("maxindex", &p(4), &[&x])[5], 2.0);
        assert_eq!(one("maxindex", &p(4), &[&x[2..]])[3], 0.0);
    }

    #[test]
    fn variance_and_stddev_hand_values() {
        // The classic window 2 4 4 4 5 5 7 9: mean 5, squared deviations sum to 32, so the
        // population variance is 32 / 8 = 4 and the deviation 2.
        let x = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let v = one("var", &p(8), &[&x]);
        assert_eq!(v[7], 4.0);
        assert!(v[..7].iter().all(|a| a.is_nan()));
        let s = one("stddev", &[ParamVal::Int(8), ParamVal::Float(1.0)], &[&x]);
        assert_eq!(s[7], 2.0);
        let s3 = one("stddev", &[ParamVal::Int(8), ParamVal::Float(3.0)], &[&x]);
        assert_eq!(s3[7], 6.0);
    }

    #[test]
    fn flat_window_has_exactly_zero_spread_even_when_the_mean_is_inexact() {
        // 0.1 + 0.1 + 0.1 = 0.30000000000000004, whose third is not 0.1.
        let x = [0.1; 8];
        assert_eq!(one("var", &p(3), &[&x])[7].to_bits(), 0.0f64.to_bits());
        let s = one("stddev", &[ParamVal::Int(3), ParamVal::Float(1.0)], &[&x]);
        assert_eq!(s[7].to_bits(), 0.0f64.to_bits());
        let slope = one("linearreg_slope", &p(3), &[&x]);
        assert_eq!(slope[7].to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn variance_recovers_after_a_nan_leaves() {
        let x = [1.0, 2.0, NAN, 4.0, 5.0, 6.0, 8.0];
        let v = one("var", &p(2), &[&x]);
        assert!(v[2].is_nan() && v[3].is_nan());
        assert_eq!(v[4], 0.25); // 4, 5: mean 4.5, deviations 0.5 each
        assert_eq!(v[6], 1.0); // 6, 8
    }

    #[test]
    fn regression_family_on_an_exact_line() {
        // y = 2x + 1 over x = 0..3: slope 2, newest-bar value 7, intercept 1, one bar ahead 9.
        let y = [1.0, 3.0, 5.0, 7.0];
        assert_eq!(one("linearreg_slope", &p(4), &[&y])[3], 2.0);
        assert_eq!(one("linearreg", &p(4), &[&y])[3], 7.0);
        assert_eq!(one("linearreg_intercept", &p(4), &[&y])[3], 1.0);
        assert_eq!(one("tsf", &p(4), &[&y])[3], 9.0);
        let angle = one("linearreg_angle", &p(4), &[&y])[3];
        assert!((angle - 2f64.atan().to_degrees()).abs() < 1e-12);
    }

    #[test]
    fn regression_on_a_noisy_window() {
        // Window 1 3 2 5, x = 0..3: mean 2.75, xbar 1.5, sum (x - xbar) y = -1.5 - 1.5 + 1 + 7.5
        // = 5.5, sxx = 5, so slope 1.1; value 2.75 + 1.1 * 1.5 = 4.4; intercept 2.75 - 1.65 = 1.1;
        // forecast 2.75 + 1.1 * 2.5 = 5.5.
        let y = [1.0, 3.0, 2.0, 5.0];
        same(&[one("linearreg_slope", &p(4), &[&y])[3]], &[1.1]);
        same(&[one("linearreg", &p(4), &[&y])[3]], &[4.4]);
        same(&[one("linearreg_intercept", &p(4), &[&y])[3]], &[1.1]);
        same(&[one("tsf", &p(4), &[&y])[3]], &[5.5]);
    }

    #[test]
    fn correl_hand_values() {
        let a = [1.0, 2.0, 3.0, 4.0];
        same(&one("correl", &p(3), &[&a, &[2.0, 4.0, 6.0, 8.0]]), &[NAN, NAN, 1.0, 1.0]);
        same(&one("correl", &p(3), &[&a, &[3.0, 2.0, 1.0, 0.0]]), &[NAN, NAN, -1.0, -1.0]);
        // A flat side has no correlation: NaN, not 0 and not a huge ratio.
        assert!(one("correl", &p(3), &[&a, &[5.0; 4]])[3].is_nan());
        // 1 2 3 against 1 3 2: sxy = 1, sxx = syy = 2, so r = 0.5.
        same(&one("correl", &p(3), &[&[1.0, 2.0, 3.0], &[1.0, 3.0, 2.0]])[2..], &[0.5]);
    }

    #[test]
    fn beta_is_defined_on_returns() {
        // x returns 0.1, -0.1, 0.1 (100 -> 110 -> 99 -> 108.9); y returns twice those
        // (100 -> 120 -> 96 -> 115.2), so beta of y on x is 2.
        let x = [100.0, 110.0, 99.0, 108.9];
        let y = [100.0, 120.0, 96.0, 115.2];
        let b = one("beta", &p(3), &[&x, &y]);
        assert!(b[..3].iter().all(|v| v.is_nan()), "first valid is bar `period`");
        assert!((b[3] - 2.0).abs() < 1e-12, "{}", b[3]);
        // Levels do not enter: scaling the first input leaves its returns, hence beta, alone.
        let x2: Vec<f64> = x.iter().map(|v| v * 7.0).collect();
        assert!((one("beta", &p(3), &[&x2, &y])[3] - 2.0).abs() < 1e-12);
        // A first input with no movement has no beta.
        assert!(one("beta", &p(3), &[&[5.0; 4], &y])[3].is_nan());
    }

    #[test]
    fn indices_recover_after_a_nan() {
        let x = [1.0, 2.0, NAN, 4.0, 3.0, 2.0, 1.0];
        let hi = one("maxindex", &p(2), &[&x]);
        assert!(hi[2].is_nan() && hi[3].is_nan());
        assert_eq!((hi[4], hi[5], hi[6]), (3.0, 4.0, 5.0));
    }
}
