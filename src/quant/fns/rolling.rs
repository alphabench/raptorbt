//! Rolling and elementwise statistics with no counterpart in TA-Lib: standardisation, moments,
//! order statistics, lags and returns, clipping, covariance, exponentially weighted moments and
//! the conditional. Each carries a hand-derived test as its reference.
//!
//! A windowed function outputs NaN while its window holds a non-finite value, then recovers; a
//! lagged one is NaN while the bar `k` back is missing or NaN.

use super::math::{fin, float_param, negligible, pair_stats, MapN};
use crate::quant::eval::ops;
use crate::quant::kernel::{Adapter, Hist, Kernel, Node};
use crate::quant::registry::{
    int_param, regs, Domain, Family, FnDef, InputDef, Invariance, Memory, NanRule, OutKind,
    ParamDef, ParamKind, ParamVal, Reference, Relation,
};
use crate::quant::window::{RollMean, RollVar};

const MAX_PERIOD: i64 = 1 << 24;

// ---------------------------------------------------------------- standardisation, moments

/// `(x - mean) / stdev` over the window, population stdev. A window whose spread is
/// negligible (flat, or under 1e-12 of its level) gives NaN, never a huge ratio.
struct ZScore {
    p: usize,
    mean: RollMean,
    var: RollVar,
}

impl Kernel<1, 1> for ZScore {
    fn lookback(&self) -> usize {
        self.p
    }

    fn step<H: Hist<1>>(&mut self, x: [f64; 1], h: &H) -> [f64; 1] {
        let m = self.mean.push(x[0], h, 0);
        let v = self.var.push(x[0], h, 0);
        if m.is_nan() || v.is_nan() || negligible(v, 1.0, m) {
            return [f64::NAN];
        }
        [fin((x[0] - m) / v.sqrt())]
    }
}

/// Mean and the sums of squared, cubed and fourth-power deviations over the window, by two
/// passes; `None` while the window is short or holds a non-finite value.
fn moments<H: Hist<1>>(p: usize, x: f64, h: &H) -> Option<(f64, f64, f64, f64)> {
    let mut sum = 0.0;
    for age in (0..p).rev() {
        let v = if age == 0 { x } else { h.ago(0, age) };
        if !v.is_finite() {
            return None;
        }
        sum += v;
    }
    let mean = sum / p as f64;
    let (mut s2, mut s3, mut s4) = (0.0, 0.0, 0.0);
    for age in (0..p).rev() {
        let v = if age == 0 { x } else { h.ago(0, age) };
        let d = v - mean;
        let d2 = d * d;
        s2 += d2;
        s3 += d2 * d;
        s4 += d2 * d2;
    }
    Some((mean, s2, s3, s4))
}

/// Biased (population) skewness `m3 / m2^1.5` or excess kurtosis `m4 / m2^2 - 3`.
struct Moment {
    p: usize,
    kurt: bool,
}

impl Kernel<1, 1> for Moment {
    fn lookback(&self) -> usize {
        self.p
    }

    fn step<H: Hist<1>>(&mut self, x: [f64; 1], h: &H) -> [f64; 1] {
        let Some((mean, s2, s3, s4)) = moments(self.p, x[0], h) else {
            return [f64::NAN];
        };
        let n = self.p as f64;
        if negligible(s2, n, mean) {
            return [f64::NAN];
        }
        let m2 = s2 / n;
        [fin(if self.kurt { (s4 / n) / (m2 * m2) - 3.0 } else { (s3 / n) / (m2 * m2.sqrt()) })]
    }
}

// ---------------------------------------------------------------- order statistics

/// The window's values in ascending order, kept by insertion and removal (O(period) moves, no
/// allocation after construction). Values equal to the bit are interchangeable, so the order
/// statistics do not depend on the path that built the window.
struct SortedWin {
    p: usize,
    v: Vec<f64>,
    bad: usize,
    n: u64,
}

impl SortedWin {
    fn new(p: usize) -> Self {
        Self { p, v: Vec::with_capacity(p), bad: 0, n: 0 }
    }

    /// Slides the window; true when it is full and holds only finite values.
    fn push<H: Hist<1>>(&mut self, x: f64, h: &H) -> bool {
        self.n += 1;
        if self.n > self.p as u64 {
            let old = h.ago(0, self.p);
            if old.is_finite() {
                let at = self.v.partition_point(|a| a.total_cmp(&old).is_lt());
                self.v.remove(at);
            } else {
                self.bad -= 1;
            }
        }
        if x.is_finite() {
            let at = self.v.partition_point(|a| a.total_cmp(&x).is_lt());
            self.v.insert(at, x);
        } else {
            self.bad += 1;
        }
        self.n >= self.p as u64 && self.bad == 0
    }
}

/// Quantile by linear interpolation between order statistics at position `q * (p - 1)`.
struct Quantile {
    w: SortedWin,
    q: f64,
}

impl Kernel<1, 1> for Quantile {
    fn lookback(&self) -> usize {
        self.w.p
    }

    fn step<H: Hist<1>>(&mut self, x: [f64; 1], h: &H) -> [f64; 1] {
        if !self.w.push(x[0], h) {
            return [f64::NAN];
        }
        let pos = self.q * (self.w.p - 1) as f64;
        let lo = pos.floor() as usize;
        let frac = pos - lo as f64;
        let s = &self.w.v;
        if frac == 0.0 || lo + 1 >= s.len() {
            return [s[lo.min(s.len() - 1)]];
        }
        [fin(s[lo] + (s[lo + 1] - s[lo]) * frac)]
    }
}

/// Time-series percentile rank: the current value's average 1-based rank among the window,
/// ties sharing their mean rank, over the window length. In (0, 1]: 1 is the highest, `1 / p`
/// the lowest.
struct RankTs {
    w: SortedWin,
}

impl Kernel<1, 1> for RankTs {
    fn lookback(&self) -> usize {
        self.w.p
    }

    fn step<H: Hist<1>>(&mut self, x: [f64; 1], h: &H) -> [f64; 1] {
        if !self.w.push(x[0], h) {
            return [f64::NAN];
        }
        let x = x[0];
        let less = self.w.v.partition_point(|a| *a < x);
        let at_most = self.w.v.partition_point(|a| *a <= x);
        let equal = (at_most - less) as f64;
        [(less as f64 + (equal + 1.0) / 2.0) / self.w.p as f64]
    }
}

/// Bars since the window's extreme: 0 when the current bar is it. A tie goes to the most
/// recent bar. Monotonic deque of (index, value) in a ring of `p` slots, amortised O(1).
struct BarsSince {
    p: usize,
    is_max: bool,
    idx: Box<[u64]>,
    val: Box<[f64]>,
    head: usize,
    len: usize,
    n: u64,
    last_bad: Option<u64>,
}

impl BarsSince {
    fn new(p: usize, is_max: bool) -> Self {
        Self {
            p,
            is_max,
            idx: vec![0; p].into_boxed_slice(),
            val: vec![f64::NAN; p].into_boxed_slice(),
            head: 0,
            len: 0,
            n: 0,
            last_bad: None,
        }
    }

    fn slot(&self, j: usize) -> usize {
        (self.head + j) % self.p
    }
}

impl Kernel<1, 1> for BarsSince {
    fn lookback(&self) -> usize {
        0
    }

    fn step<H: Hist<1>>(&mut self, x: [f64; 1], _h: &H) -> [f64; 1] {
        let x = x[0];
        let cur = self.n;
        self.n += 1;
        while self.len > 0 && cur - self.idx[self.head] >= self.p as u64 {
            self.head = self.slot(1);
            self.len -= 1;
        }
        if x.is_finite() {
            while self.len > 0 {
                let back = self.val[self.slot(self.len - 1)];
                if (self.is_max && back <= x) || (!self.is_max && back >= x) {
                    self.len -= 1;
                } else {
                    break;
                }
            }
            let at = self.slot(self.len);
            self.idx[at] = cur;
            self.val[at] = x;
            self.len += 1;
        } else {
            self.last_bad = Some(cur);
        }
        if self.n < self.p as u64 || self.last_bad.is_some_and(|b| cur - b < self.p as u64) {
            return [f64::NAN];
        }
        [(cur - self.idx[self.head]) as f64]
    }
}

// ---------------------------------------------------------------- lags and returns

#[derive(Clone, Copy)]
enum Lag {
    Shift,
    Diff,
    Pct,
    Log,
}

/// A function of the current value and the one `k` bars back.
struct Lagged {
    k: usize,
    kind: Lag,
}

impl Kernel<1, 1> for Lagged {
    fn lookback(&self) -> usize {
        self.k
    }

    fn step<H: Hist<1>>(&mut self, x: [f64; 1], h: &H) -> [f64; 1] {
        let (x, old) = (x[0], h.ago(0, self.k));
        [match self.kind {
            Lag::Shift => fin(old),
            Lag::Diff => fin(x - old),
            Lag::Pct => fin(ops::div(x, old) - 1.0),
            // Both prices must be positive: a ratio of two negatives is positive but is no return.
            Lag::Log if x > 0.0 && old > 0.0 => ops::ln(ops::div(x, old)),
            Lag::Log => f64::NAN,
        }]
    }
}

struct Clip {
    lo: f64,
    hi: f64,
}

impl Kernel<1, 1> for Clip {
    fn lookback(&self) -> usize {
        0
    }

    fn step<H: Hist<1>>(&mut self, x: [f64; 1], _h: &H) -> [f64; 1] {
        [if x[0].is_nan() { f64::NAN } else { x[0].clamp(self.lo, self.hi) }]
    }
}

// ---------------------------------------------------------------- covariance, ewm

/// Population covariance of the two inputs' levels over the window.
struct Cov {
    p: usize,
}

impl Kernel<2, 1> for Cov {
    fn lookback(&self) -> usize {
        self.p
    }

    fn step<H: Hist<2>>(&mut self, x: [f64; 2], h: &H) -> [f64; 1] {
        [pair_stats(self.p, x, h, false).map_or(f64::NAN, |s| fin(s.sxy / s.n))]
    }
}

/// Exponentially weighted mean, `adjust=False` form: seeded with the first value,
/// `m += alpha * (x - m)`. A NaN neither seeds nor advances it.
struct EwmMean {
    alpha: f64,
    m: Option<f64>,
}

impl Kernel<1, 1> for EwmMean {
    fn lookback(&self) -> usize {
        0
    }

    fn step<H: Hist<1>>(&mut self, x: [f64; 1], _h: &H) -> [f64; 1] {
        let x = x[0];
        if !x.is_finite() {
            return [f64::NAN];
        }
        let m = match self.m {
            Some(prev) => prev + self.alpha * (x - prev),
            None => x,
        };
        self.m = Some(m);
        [fin(m)]
    }
}

/// Exponentially weighted variance about the weighted mean (biased): with `d = x - m`,
/// `m += alpha * d` and `var = (1 - alpha) * (var + alpha * d * d)`. Zero at the first value.
struct EwmVar {
    alpha: f64,
    state: Option<(f64, f64)>,
}

impl Kernel<1, 1> for EwmVar {
    fn lookback(&self) -> usize {
        0
    }

    fn step<H: Hist<1>>(&mut self, x: [f64; 1], _h: &H) -> [f64; 1] {
        let x = x[0];
        if !x.is_finite() {
            return [f64::NAN];
        }
        let (m, v) = match self.state {
            Some((m, v)) => {
                let d = x - m;
                (m + self.alpha * d, (1.0 - self.alpha) * (v + self.alpha * d * d))
            }
            None => (x, 0.0),
        };
        self.state = Some((m, v));
        [fin(v)]
    }
}

fn ewm_settle(p: &[ParamVal], eps: f64) -> usize {
    // Seed influence decays as (1 - alpha)^k.
    let alpha = float_param(p, 0).unwrap_or(1.0);
    if alpha >= 1.0 {
        return 0;
    }
    (eps.ln() / (1.0 - alpha).ln()).ceil().max(0.0) as usize
}

// ---------------------------------------------------------------- makers

fn period_of(p: &[ParamVal]) -> Result<usize, String> {
    let v = int_param(p, 0)?;
    if v == 0 {
        return Err("period must be at least 1".into());
    }
    Ok(v)
}

fn make_zscore(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let p = period_of(p)?;
    let k = ZScore { p, mean: RollMean::new(p), var: RollVar::new(p, 0) };
    Ok(Box::new(Adapter::new(
        k,
        regs::<1>(ins, "zscore inputs")?,
        regs::<1>(outs, "zscore outputs")?,
    )))
}

fn make_skew(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = Moment { p: period_of(p)?, kurt: false };
    Ok(Box::new(Adapter::new(k, regs::<1>(ins, "skew inputs")?, regs::<1>(outs, "skew outputs")?)))
}

fn make_kurt(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = Moment { p: period_of(p)?, kurt: true };
    Ok(Box::new(Adapter::new(k, regs::<1>(ins, "kurt inputs")?, regs::<1>(outs, "kurt outputs")?)))
}

fn make_quantile(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = Quantile { w: SortedWin::new(period_of(p)?), q: float_param(p, 1)? };
    Ok(Box::new(Adapter::new(
        k,
        regs::<1>(ins, "quantile inputs")?,
        regs::<1>(outs, "quantile outputs")?,
    )))
}

fn make_rank_ts(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = RankTs { w: SortedWin::new(period_of(p)?) };
    Ok(Box::new(Adapter::new(
        k,
        regs::<1>(ins, "rank_ts inputs")?,
        regs::<1>(outs, "rank_ts outputs")?,
    )))
}

fn make_bars_since_max(
    p: &[ParamVal],
    ins: &[usize],
    outs: &[usize],
) -> Result<Box<dyn Node>, String> {
    let k = BarsSince::new(period_of(p)?, true);
    Ok(Box::new(Adapter::new(
        k,
        regs::<1>(ins, "bars_since_max inputs")?,
        regs::<1>(outs, "bars_since_max outputs")?,
    )))
}

fn make_bars_since_min(
    p: &[ParamVal],
    ins: &[usize],
    outs: &[usize],
) -> Result<Box<dyn Node>, String> {
    let k = BarsSince::new(period_of(p)?, false);
    Ok(Box::new(Adapter::new(
        k,
        regs::<1>(ins, "bars_since_min inputs")?,
        regs::<1>(outs, "bars_since_min outputs")?,
    )))
}

macro_rules! make_lag {
    ($fname:ident, $kind:expr, $label:literal) => {
        fn $fname(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
            let k = Lagged { k: period_of(p)?, kind: $kind };
            Ok(Box::new(Adapter::new(
                k,
                regs::<1>(ins, concat!($label, " inputs"))?,
                regs::<1>(outs, concat!($label, " outputs"))?,
            )))
        }
    };
}
make_lag!(make_shift, Lag::Shift, "shift");
make_lag!(make_diff, Lag::Diff, "diff");
make_lag!(make_pct_change, Lag::Pct, "pct_change");
make_lag!(make_log_return, Lag::Log, "log_return");

fn make_abs(_p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = MapN::<1>(|x| fin(x[0].abs()));
    Ok(Box::new(Adapter::new(k, regs::<1>(ins, "abs inputs")?, regs::<1>(outs, "abs outputs")?)))
}

fn make_sign(_p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = MapN::<1>(|x| {
        if x[0].is_nan() {
            f64::NAN
        } else if x[0] > 0.0 {
            1.0
        } else if x[0] < 0.0 {
            -1.0
        } else {
            0.0
        }
    });
    Ok(Box::new(Adapter::new(k, regs::<1>(ins, "sign inputs")?, regs::<1>(outs, "sign outputs")?)))
}

fn make_clip(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let (lo, hi) = (float_param(p, 0)?, float_param(p, 1)?);
    if lo > hi {
        return Err(format!("clip needs lo <= hi, got lo {lo} and hi {hi}"));
    }
    Ok(Box::new(Adapter::new(
        Clip { lo, hi },
        regs::<1>(ins, "clip inputs")?,
        regs::<1>(outs, "clip outputs")?,
    )))
}

fn make_cov(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = Cov { p: period_of(p)? };
    Ok(Box::new(Adapter::new(k, regs::<2>(ins, "cov inputs")?, regs::<1>(outs, "cov outputs")?)))
}

fn make_ewm_mean(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = EwmMean { alpha: float_param(p, 0)?, m: None };
    Ok(Box::new(Adapter::new(
        k,
        regs::<1>(ins, "ewm_mean inputs")?,
        regs::<1>(outs, "ewm_mean outputs")?,
    )))
}

fn make_ewm_var(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = EwmVar { alpha: float_param(p, 0)?, state: None };
    Ok(Box::new(Adapter::new(
        k,
        regs::<1>(ins, "ewm_var inputs")?,
        regs::<1>(outs, "ewm_var outputs")?,
    )))
}

/// Kleene: an unknown condition is unknown, whatever the branches hold.
fn make_if_else(_p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = MapN::<3>(|x| {
        if x[0].is_nan() {
            f64::NAN
        } else if x[0] != 0.0 {
            fin(x[1])
        } else {
            fin(x[2])
        }
    });
    Ok(Box::new(Adapter::new(
        k,
        regs::<3>(ins, "if_else inputs")?,
        regs::<1>(outs, "if_else outputs")?,
    )))
}

// ---------------------------------------------------------------- the table

const IN_X: &[InputDef] = &[InputDef { name: "x", default_field: Some("close") }];
const IN_AB: &[InputDef] = &[
    InputDef { name: "a", default_field: Some("high") },
    InputDef { name: "b", default_field: Some("low") },
];
const IN_COND: &[InputDef] = &[
    InputDef { name: "cond", default_field: None },
    InputDef { name: "a", default_field: None },
    InputDef { name: "b", default_field: None },
];

const fn period(min: i64) -> ParamDef {
    ParamDef { name: "period", kind: ParamKind::Int { min, max: MAX_PERIOD }, default: None }
}

const PERIOD_2: &[ParamDef] = &[period(2)];
const PERIOD_3: &[ParamDef] = &[period(3)];
const PERIOD_4: &[ParamDef] = &[period(4)];
const PERIOD_1: &[ParamDef] = &[period(1)];
const QUANTILE_PARAMS: &[ParamDef] = &[
    period(1),
    ParamDef {
        name: "q",
        kind: ParamKind::Float { min: 0.0, max: 1.0 },
        default: Some(ParamVal::Float(0.5)),
    },
];
const LAG: &[ParamDef] = &[ParamDef {
    name: "k",
    kind: ParamKind::Int { min: 1, max: MAX_PERIOD },
    default: Some(ParamVal::Int(1)),
}];
const CLIP_PARAMS: &[ParamDef] = &[
    ParamDef { name: "lo", kind: ParamKind::Float { min: -1e300, max: 1e300 }, default: None },
    ParamDef { name: "hi", kind: ParamKind::Float { min: -1e300, max: 1e300 }, default: None },
];
const ALPHA: &[ParamDef] = &[ParamDef {
    name: "alpha",
    kind: ParamKind::Float { min: 1e-6, max: 1.0 },
    default: Some(ParamVal::Float(0.1)),
}];

fn warm_none(_: &[ParamVal]) -> usize {
    0
}

fn warm_period(p: &[ParamVal]) -> usize {
    int_param(p, 0).map_or(0, |v| v.saturating_sub(1))
}

fn warm_lag(p: &[ParamVal]) -> usize {
    int_param(p, 0).unwrap_or(0)
}

const SCALE_SHIFT_EQ: &[Invariance] = &[Invariance::ScaleEquivariant, Invariance::ShiftEquivariant];
const SCALE_SHIFT_INV: &[Invariance] = &[Invariance::ScaleInvariant, Invariance::ShiftInvariant];

/// A function with no outside library: its reference is the named hand-derived test.
macro_rules! nolib {
    ($name:literal, $inputs:expr, $params:expr, $warm:expr, $memory:expr, $doc:expr, $inv:expr,
     $nan:expr, $domain:expr, $hand:literal, $make:expr) => {
        FnDef {
            name: $name,
            family: Family::Rolling,
            inputs: $inputs,
            params: $params,
            outputs: &["value"],
            out_kind: OutKind::Series,
            warmup: $warm,
            memory: $memory,
            legacy: None,
            talib_name: None,
            doc: $doc,
            refs: &[],
            rev: 1,
            invariance: $inv,
            causal: true,
            nan_rule: $nan,
            domain: $domain,
            reference: Reference::Hand(concat!("fns::rolling::tests::", $hand)),
            relation: Relation::NoLibrary,
            make: $make,
        }
    };
}

pub const DEFS: &[FnDef] = &[
    nolib!("zscore", IN_X, PERIOD_2, warm_period, Memory::Finite,
        "(x - mean) / stdev over the last `period` values, population stdev. NaN when the window is flat or its spread is under 1e-12 of its level.",
        SCALE_SHIFT_INV, NanRule::Windowed, Domain::Any, "zscore_hand_value", make_zscore),
    nolib!("skew", IN_X, PERIOD_3, warm_period, Memory::Finite,
        "Biased (population) skewness of the last `period` values: m3 / m2^1.5. NaN for a flat window.",
        SCALE_SHIFT_INV, NanRule::Windowed, Domain::Any, "skew_kurt_hand_values", make_skew),
    nolib!("kurt", IN_X, PERIOD_4, warm_period, Memory::Finite,
        "Biased (population) excess kurtosis of the last `period` values: m4 / m2^2 - 3. NaN for a flat window.",
        SCALE_SHIFT_INV, NanRule::Windowed, Domain::Any, "skew_kurt_hand_values", make_kurt),
    nolib!("quantile", IN_X, QUANTILE_PARAMS, warm_period, Memory::Finite,
        "Quantile `q` in [0, 1] of the last `period` values, linear interpolation between order statistics at position q * (period - 1). q = 0 is the minimum, 1 the maximum, 0.5 the median.",
        SCALE_SHIFT_EQ, NanRule::Windowed, Domain::Any, "quantile_hand_values", make_quantile),
    nolib!("rank_ts", IN_X, PERIOD_1, warm_period, Memory::Finite,
        "Time-series percentile rank of the current value in the last `period` values: its average 1-based rank, ties sharing the mean rank, divided by `period`. In (0, 1]: 1 when it is the highest, 1 / period when the lowest.",
        SCALE_SHIFT_INV, NanRule::Windowed, Domain::Any, "rank_ts_hand_values", make_rank_ts),
    nolib!("shift", IN_X, LAG, warm_lag, Memory::Finite,
        "The value `k` bars ago (k >= 1). There is no negative shift: the future cannot be addressed.",
        SCALE_SHIFT_EQ, NanRule::Windowed, Domain::Any, "lag_functions_hand_values", make_shift),
    nolib!("diff", IN_X, LAG, warm_lag, Memory::Finite,
        "x - x[k bars ago].",
        &[Invariance::ScaleEquivariant, Invariance::ShiftInvariant], NanRule::Windowed,
        Domain::Any, "lag_functions_hand_values", make_diff),
    nolib!("pct_change", IN_X, LAG, warm_lag, Memory::Finite,
        "x / x[k bars ago] - 1; NaN when the earlier value is 0.",
        &[Invariance::ScaleInvariant], NanRule::Windowed, Domain::Any,
        "lag_functions_hand_values", make_pct_change),
    nolib!("log_return", IN_X, LAG, warm_lag, Memory::Finite,
        "ln(x / x[k bars ago]); NaN unless both prices are positive.",
        &[Invariance::ScaleInvariant], NanRule::Windowed, Domain::Positive,
        "lag_functions_hand_values", make_log_return),
    nolib!("abs", IN_X, &[], warm_none, Memory::Finite, "Absolute value.",
        &[Invariance::ScaleEquivariant], NanRule::Elementwise, Domain::Any,
        "abs_sign_clip_hand_values", make_abs),
    nolib!("sign", IN_X, &[], warm_none, Memory::Finite, "-1, 0 or 1 by the sign of x.",
        &[Invariance::ScaleInvariant], NanRule::Elementwise, Domain::Any,
        "abs_sign_clip_hand_values", make_sign),
    nolib!("clip", IN_X, CLIP_PARAMS, warm_none, Memory::Finite,
        "x limited to [lo, hi]; lo must not exceed hi.",
        &[], NanRule::Elementwise, Domain::Any, "abs_sign_clip_hand_values", make_clip),
    nolib!("cov", IN_AB, PERIOD_1, warm_period, Memory::Finite,
        "Population covariance of the two inputs' levels over the last `period` values (divisor `period`). 0 when either is flat.",
        &[Invariance::ShiftInvariant], NanRule::Windowed, Domain::Any, "cov_hand_value", make_cov),
    nolib!("ewm_mean", IN_X, ALPHA, warm_none, Memory::Infinite { settle: ewm_settle },
        "Exponentially weighted mean with smoothing factor `alpha` in (0, 1], seeded with the first value: m += alpha * (x - m). Unlike `ema` it takes alpha, not a period, and has no averaging seed. A NaN holds the state.",
        SCALE_SHIFT_EQ, NanRule::Recursive, Domain::Any, "ewm_hand_values", make_ewm_mean),
    nolib!("ewm_var", IN_X, ALPHA, warm_none, Memory::Infinite { settle: ewm_settle },
        "Exponentially weighted variance (biased) about the weighted mean, factor `alpha`: with d = x - m, m += alpha * d and var = (1 - alpha) * (var + alpha * d * d). 0 at the first value. A NaN holds the state.",
        &[Invariance::ShiftInvariant], NanRule::Recursive, Domain::Any, "ewm_hand_values",
        make_ewm_var),
    nolib!("if_else", IN_COND, &[], warm_none, Memory::Finite,
        "a where `cond` is true (non-zero), b where false (zero), NaN where `cond` is unknown (NaN) whatever a and b hold. A NaN in the chosen branch passes through.",
        &[Invariance::ScaleEquivariant], NanRule::Elementwise, Domain::Any, "if_else_kleene",
        make_if_else),
    nolib!("bars_since_max", IN_X, PERIOD_1, warm_period, Memory::Finite,
        "Bars since the highest value of the last `period` values: 0 when the current bar is the high. A tie goes to the most recent bar. The causal, relative form of maxindex.",
        SCALE_SHIFT_INV, NanRule::Windowed, Domain::Any, "bars_since_hand_values",
        make_bars_since_max),
    nolib!("bars_since_min", IN_X, PERIOD_1, warm_period, Memory::Finite,
        "Bars since the lowest value of the last `period` values: 0 when the current bar is the low. A tie goes to the most recent bar. The causal, relative form of minindex.",
        SCALE_SHIFT_INV, NanRule::Windowed, Domain::Any, "bars_since_hand_values",
        make_bars_since_min),
];

#[cfg(test)]
mod tests {
    use super::super::math::testkit::*;
    use super::*;

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
    fn every_rolling_function_agrees_between_batch_and_streaming() {
        check_table(DEFS, 32);
    }

    #[test]
    fn every_hand_reference_names_a_test_in_this_file() {
        let source = include_str!("rolling.rs");
        for d in DEFS {
            let Reference::Hand(path) = d.reference else { panic!("{} has no hand test", d.name) };
            let name = path.rsplit("::").next().unwrap();
            assert!(source.contains(&format!("fn {name}(")), "{}: no test {name}", d.name);
            assert_eq!(d.relation, Relation::NoLibrary);
            assert!(d.talib_name.is_none());
        }
        assert_eq!(DEFS.len(), 19);
    }

    #[test]
    fn zscore_hand_value() {
        // Window 1 2 3: mean 2, population variance 2/3, so z of the newest = 1 / sqrt(2/3).
        let z = one("zscore", &p(3), &[&[1.0, 2.0, 3.0]]);
        assert!(z[0].is_nan() && z[1].is_nan());
        same(&z[2..], &[1.5f64.sqrt()]);
        // A flat window has no spread to divide by.
        assert!(one("zscore", &p(3), &[&[4.0; 5]])[4].is_nan());
    }

    #[test]
    fn skew_kurt_hand_values() {
        // Window 0 0 0 4: mean 1, deviations -1 -1 -1 3, m2 = 12/4 = 3, m3 = 24/4 = 6,
        // m4 = 84/4 = 21. Skew = 6 / 3^1.5 = 2/sqrt(3); excess kurtosis = 21/9 - 3 = -2/3.
        let x = [0.0, 0.0, 0.0, 4.0];
        same(&one("skew", &p(4), &[&x])[3..], &[2.0 / 3f64.sqrt()]);
        same(&one("kurt", &p(4), &[&x])[3..], &[-2.0 / 3.0]);
        // A symmetric window has skew 0.
        assert_eq!(one("skew", &p(3), &[&[1.0, 2.0, 3.0]])[2], 0.0);
        assert!(one("kurt", &p(4), &[&[2.0; 6]])[5].is_nan());
    }

    #[test]
    fn quantile_hand_values() {
        // Window 4 1 3 2 sorts to 1 2 3 4; position q * 3. q = 0.75 -> 2.25 -> 3 + 0.25 * (4 - 3).
        let x = [4.0, 1.0, 3.0, 2.0, 10.0];
        let q = |q: f64| one("quantile", &[ParamVal::Int(4), ParamVal::Float(q)], &[&x]);
        same(&q(0.75)[3..], &[3.25, 4.75]); // next window 1 3 2 10 sorts to 1 2 3 10
        same(&q(0.0)[3..], &[1.0, 1.0]);
        same(&q(1.0)[3..], &[4.0, 10.0]);
        same(&q(0.5)[3..], &[2.5, 2.5]);
    }

    #[test]
    fn rank_ts_hand_values() {
        // [1 2 3]: 3 is the highest, rank 3 of 3. [2 3 2]: the two 2s share ranks 1 and 2,
        // mean 1.5, over 3. [3 2 2]: likewise.
        let r = one("rank_ts", &p(3), &[&[1.0, 2.0, 3.0, 2.0, 2.0]]);
        same(&r[2..], &[1.0, 0.5, 0.5]);
        // The lowest of a window is 1 / period.
        same(&one("rank_ts", &p(4), &[&[5.0, 6.0, 7.0, 1.0]])[3..], &[0.25]);
    }

    #[test]
    fn lag_functions_hand_values() {
        let x = [1.0, 2.0, 4.0, 8.0];
        let k = |v: i64| vec![ParamVal::Int(v)];
        same(&one("shift", &k(2), &[&x]), &[NAN, NAN, 1.0, 2.0]);
        same(&one("diff", &k(1), &[&x]), &[NAN, 1.0, 2.0, 4.0]);
        same(&one("pct_change", &k(1), &[&x]), &[NAN, 1.0, 1.0, 1.0]);
        same(&one("log_return", &k(1), &[&x]), &[NAN, 2f64.ln(), 2f64.ln(), 2f64.ln()]);
        // An earlier 0 has no percentage change; non-positive prices have no log return.
        assert!(one("pct_change", &k(1), &[&[0.0, 5.0]])[1].is_nan());
        let lr = one("log_return", &k(1), &[&[-1.0, -2.0, 3.0, 6.0]]);
        assert!(lr[1].is_nan() && lr[2].is_nan());
        same(&lr[3..], &[2f64.ln()]);
    }

    #[test]
    fn abs_sign_clip_hand_values() {
        let x = [-2.0, 0.0, 3.0, NAN];
        same(&one("abs", &[], &[&x]), &[2.0, 0.0, 3.0, NAN]);
        same(&one("sign", &[], &[&x]), &[-1.0, 0.0, 1.0, NAN]);
        let clip = [ParamVal::Float(-1.0), ParamVal::Float(1.0)];
        same(&one("clip", &clip, &[&[-2.0, 0.5, 3.0, NAN]]), &[-1.0, 0.5, 1.0, NAN]);
        let def = crate::quant::registry::lookup("clip").unwrap();
        let bad = [ParamVal::Float(2.0), ParamVal::Float(1.0)];
        assert!((def.make)(&bad, &[0], &[1]).is_err(), "lo above hi is refused");
    }

    #[test]
    fn cov_hand_value() {
        // Means 2 and 4, deviations -1 0 1 and -2 0 2: sum of products 4, over 3.
        let c = one("cov", &p(3), &[&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]]);
        same(&c[2..], &[4.0 / 3.0]);
        assert_eq!(
            one("cov", &p(3), &[&[1.0, 2.0, 3.0], &[5.0; 3]])[2].to_bits(),
            0.0f64.to_bits()
        );
    }

    #[test]
    fn ewm_hand_values() {
        // alpha 0.5 on 2 4 8: mean 2, 2 + 0.5 * 2 = 3, 3 + 0.5 * 5 = 5.5. Variance 0, then
        // d = 2: 0.5 * (0 + 0.5 * 4) = 1, then d = 5: 0.5 * (1 + 0.5 * 25) = 6.75, which is the
        // weighted variance about 5.5 with weights 0.25, 0.25, 0.5: 3.0625 + 0.5625 + 3.125.
        let a = [ParamVal::Float(0.5)];
        let x = [2.0, 4.0, 8.0];
        same(&one("ewm_mean", &a, &[&x]), &[2.0, 3.0, 5.5]);
        same(&one("ewm_var", &a, &[&x]), &[0.0, 1.0, 6.75]);
        // A NaN holds the state; the next value continues from it.
        same(&one("ewm_mean", &a, &[&[2.0, NAN, 4.0]]), &[2.0, NAN, 3.0]);
        let def = crate::quant::registry::lookup("ewm_mean").unwrap();
        let crate::quant::registry::Memory::Infinite { settle } = def.memory else { panic!() };
        // 0.5^k below 1e-3 first at k = 10.
        assert_eq!(settle(&a, 1e-3), 10);
    }

    #[test]
    fn if_else_kleene() {
        let cond = [1.0, 0.0, NAN, 2.0, 1.0];
        let a = [10.0, 10.0, 10.0, 10.0, NAN];
        let b = [20.0; 5];
        same(&one("if_else", &[], &[&cond, &a, &b]), &[10.0, 20.0, NAN, 10.0, NAN]);
        // An unknown condition stays unknown even when both branches agree.
        assert!(one("if_else", &[], &[&[NAN], &[7.0], &[7.0]])[0].is_nan());
    }

    #[test]
    fn bars_since_hand_values() {
        // Windows of 3 over 1 3 2 5 4 4. Highs sit at bars 1, 3, 3, 3 -> 1, 0, 1, 2 bars ago.
        // Lows: 1 at bar 0, 2 at bar 2, 2 at bar 2, and 4 at bars 4 and 5 where the tie goes to
        // the newer one -> 2, 1, 2, 0.
        let x = [1.0, 3.0, 2.0, 5.0, 4.0, 4.0];
        same(&one("bars_since_max", &p(3), &[&x]), &[NAN, NAN, 1.0, 0.0, 1.0, 2.0]);
        same(&one("bars_since_min", &p(3), &[&x]), &[NAN, NAN, 2.0, 1.0, 2.0, 0.0]);
    }
}
