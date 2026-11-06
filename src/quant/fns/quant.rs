//! Quant extras: what TA-Lib lacks and quants need. Volatility estimators, rolling regression,
//! mean-reversion and memory measures, window tail risk, price structure.
//!
//! Almost every function here is a pure function of the last `w` bars, evaluated by one generic
//! kernel (`Win`): it gathers the window from `Hist` into preallocated buffers and calls the
//! function, so batch and streaming agree bit for bit by construction. The price is O(w) per
//! bar (more where a sort is needed; each `doc` says). A NaN or infinity anywhere in the window
//! gives NaN until it leaves. Annualisation, risk-free rates and sign conventions beyond those
//! stated in a `doc` are the caller's.

use crate::quant::kernel::{Adapter, Hist, Kernel, Node};
use crate::quant::registry::{
    int_param, regs, Domain, Family, FnDef, InputDef, Invariance, Memory, NanRule, OutKind,
    ParamDef, ParamKind, ParamVal, Reference, Relation,
};
use crate::quant::window::RollExt;
use std::array;
use std::f64::consts::LN_2;

const MAX_PERIOD: i64 = 1 << 24;
/// Longest fractional-differencing window; a threshold that needs more is refused at build.
const MAX_FFD: usize = 1 << 20;
/// A centred sum of squares at or below this fraction of the raw sum of squares is rounding
/// noise: the window is flat for every practical purpose and a ratio over it would be huge.
const DEGENERATE: f64 = 1e-12;

// ---------------------------------------------------------------------------------------------
// The generic window kernel.

/// What a window function reads besides its inputs.
pub(crate) struct Cfg {
    pub p: usize,
    pub a: f64,
    pub b: f64,
    pub w: Vec<f64>,
}

impl Cfg {
    fn new(p: usize) -> Self {
        Self { p, a: 0.0, b: 0.0, w: Vec::new() }
    }
}

/// Preallocated working space, `max window` long each.
pub(crate) struct Scratch {
    pub a: Vec<f64>,
    pub b: Vec<f64>,
    pub idx: Vec<usize>,
}

/// A window function: input `j` is the last `lens[j]` values, oldest first, current last.
pub(crate) type WinFn<const I: usize> = fn(&[&[f64]; I], &Cfg, &mut Scratch) -> f64;

pub(crate) struct Win<const I: usize> {
    lens: [usize; I],
    cfg: Cfg,
    f: WinFn<I>,
    bufs: [Vec<f64>; I],
    scratch: Scratch,
}

impl<const I: usize> Win<I> {
    pub(crate) fn new(lens: [usize; I], cfg: Cfg, f: WinFn<I>) -> Self {
        let n = lens.iter().copied().max().unwrap_or(1).max(1);
        Self {
            lens,
            cfg,
            f,
            bufs: array::from_fn(|j| vec![f64::NAN; lens[j]]),
            scratch: Scratch { a: vec![0.0; n], b: vec![0.0; n], idx: vec![0; n] },
        }
    }
}

impl<const I: usize> Kernel<I, 1> for Win<I> {
    fn heap_bytes(&self) -> usize {
        self.bufs.iter().map(|b| b.capacity() * std::mem::size_of::<f64>()).sum::<usize>()
            + self.cfg.w.capacity() * std::mem::size_of::<f64>()
            + self.scratch.a.capacity() * std::mem::size_of::<f64>()
            + self.scratch.b.capacity() * std::mem::size_of::<f64>()
            + self.scratch.idx.capacity() * std::mem::size_of::<usize>()
    }

    fn lookback(&self) -> usize {
        self.lens.iter().copied().max().unwrap_or(1) - 1
    }

    #[inline(always)]
    fn step<H: Hist<I>>(&mut self, x: [f64; I], h: &H) -> [f64; 1] {
        let mut ok = true;
        for (j, &xj) in x.iter().enumerate() {
            let l = self.lens[j];
            let buf = &mut self.bufs[j];
            buf[l - 1] = xj;
            ok &= xj.is_finite();
            for k in 1..l {
                let v = h.ago(j, k);
                ok &= v.is_finite();
                buf[l - 1 - k] = v;
            }
        }
        if !ok {
            return [f64::NAN];
        }
        let views: [&[f64]; I] = array::from_fn(|j| &self.bufs[j][..]);
        let v = (self.f)(&views, &self.cfg, &mut self.scratch);
        [if v.is_finite() { v } else { f64::NAN }]
    }
}

fn mk<const I: usize>(
    lens: [usize; I],
    cfg: Cfg,
    f: WinFn<I>,
    ins: &[usize],
    outs: &[usize],
) -> Result<Box<dyn Node>, String> {
    let k = Win::new(lens, cfg, f);
    Ok(Box::new(Adapter::new(k, regs::<I>(ins, "inputs")?, regs::<1>(outs, "outputs")?)))
}

fn period(params: &[ParamVal], min: usize) -> Result<usize, String> {
    let p = int_param(params, 0)?;
    if p < min {
        return Err(format!("period must be at least {min}, got {p}"));
    }
    Ok(p)
}

fn float_param(params: &[ParamVal], i: usize) -> Result<f64, String> {
    match params.get(i) {
        Some(ParamVal::Float(v)) if v.is_finite() => Ok(*v),
        Some(ParamVal::Int(v)) => Ok(*v as f64),
        other => Err(format!("parameter {i} must be a finite number, got {other:?}")),
    }
}

// ---------------------------------------------------------------------------------------------
// Statistics helpers over one window.

fn sum(x: &[f64]) -> f64 {
    x.iter().sum()
}

fn degenerate(ss: f64, raw: f64) -> bool {
    ss.is_nan() || raw.is_nan() || ss <= DEGENERATE * raw
}

/// Centred and raw sums of squares.
fn spread(x: &[f64]) -> (f64, f64, f64) {
    let m = sum(x) / x.len() as f64;
    let (mut ss, mut raw) = (0.0, 0.0);
    for &e in x {
        ss += (e - m) * (e - m);
        raw += e * e;
    }
    (m, ss, raw)
}

/// Two-pass sample variance (divisor n - 1).
fn sample_var(x: &[f64]) -> f64 {
    spread(x).1 / (x.len() - 1) as f64
}

fn all_positive(cols: &[&[f64]]) -> bool {
    cols.iter().all(|c| c.iter().all(|&v| v > 0.0))
}

struct Fit {
    beta: f64,
    alpha: f64,
    r2: f64,
}

/// OLS of `y` on `x`. `None` when `x` is flat relative to its scale. `r2` is NaN when `y` is.
fn fit(y: &[f64], x: &[f64]) -> Option<Fit> {
    let n = x.len() as f64;
    let mx = sum(x) / n;
    let my = sum(y) / n;
    let (mut sxx, mut sxy, mut syy, mut rx, mut ry) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (&xi, &yi) in x.iter().zip(y) {
        let (dx, dy) = (xi - mx, yi - my);
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
        rx += xi * xi;
        ry += yi * yi;
    }
    if degenerate(sxx, rx) {
        return None;
    }
    let beta = sxy / sxx;
    let r2 = if degenerate(syy, ry) { f64::NAN } else { (sxy * sxy / (sxx * syy)).min(1.0) };
    Some(Fit { beta, alpha: my - beta * mx, r2 })
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = sum(x) / n;
    let my = sum(y) / n;
    let (mut sxx, mut sxy, mut syy, mut rx, mut ry) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (&xi, &yi) in x.iter().zip(y) {
        let (dx, dy) = (xi - mx, yi - my);
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
        rx += xi * xi;
        ry += yi * yi;
    }
    if degenerate(sxx, rx) || degenerate(syy, ry) {
        return f64::NAN;
    }
    (sxy / (sxx * syy).sqrt()).clamp(-1.0, 1.0)
}

/// Average ranks (ties share the mean of their positions), 1-based.
fn ranks(v: &[f64], idx: &mut [usize], out: &mut [f64]) {
    let n = v.len();
    for (i, slot) in idx.iter_mut().enumerate().take(n) {
        *slot = i;
    }
    idx[..n].sort_unstable_by(|&i, &j| v[i].total_cmp(&v[j]));
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && v[idx[j + 1]] == v[idx[i]] {
            j += 1;
        }
        let r = (i + j) as f64 / 2.0 + 1.0;
        for &k in &idx[i..=j] {
            out[k] = r;
        }
        i = j + 1;
    }
}

// ---------------------------------------------------------------------------------------------
// Volatility estimators. All return a per-bar volatility (square root of the variance of the
// log return of one bar); annualising is the caller's multiplication.

fn f_vol_close(v: &[&[f64]; 1], c: &Cfg, s: &mut Scratch) -> f64 {
    let px = v[0];
    if !all_positive(&[px]) {
        return f64::NAN;
    }
    let r = &mut s.a[..c.p];
    for (slot, w) in r.iter_mut().zip(px.windows(2)) {
        *slot = (w[1] / w[0]).ln();
    }
    sample_var(r).sqrt()
}

fn f_vol_parkinson(v: &[&[f64]; 2], c: &Cfg, _s: &mut Scratch) -> f64 {
    let (h, l) = (v[0], v[1]);
    if !all_positive(&[h, l]) {
        return f64::NAN;
    }
    let mut acc = 0.0;
    for (hi, lo) in h.iter().zip(l) {
        let u = (hi / lo).ln();
        acc += u * u;
    }
    (acc / (c.p as f64 * 4.0 * LN_2)).sqrt()
}

fn sqrt_or_nan(var: f64) -> f64 {
    if var < 0.0 {
        f64::NAN
    } else {
        var.sqrt()
    }
}

fn f_vol_garman_klass(v: &[&[f64]; 4], c: &Cfg, _s: &mut Scratch) -> f64 {
    let (o, h, l, cl) = (v[0], v[1], v[2], v[3]);
    if !all_positive(&[o, h, l, cl]) {
        return f64::NAN;
    }
    let k = 2.0 * LN_2 - 1.0;
    let mut acc = 0.0;
    for i in 0..c.p {
        let u = (h[i] / l[i]).ln();
        let d = (cl[i] / o[i]).ln();
        acc += 0.5 * u * u - k * d * d;
    }
    sqrt_or_nan(acc / c.p as f64)
}

fn rs_term(o: f64, h: f64, l: f64, c: f64) -> f64 {
    (h / c).ln() * (h / o).ln() + (l / c).ln() * (l / o).ln()
}

fn f_vol_rogers_satchell(v: &[&[f64]; 4], c: &Cfg, _s: &mut Scratch) -> f64 {
    let (o, h, l, cl) = (v[0], v[1], v[2], v[3]);
    if !all_positive(&[o, h, l, cl]) {
        return f64::NAN;
    }
    let acc: f64 = (0..c.p).map(|i| rs_term(o[i], h[i], l[i], cl[i])).sum();
    sqrt_or_nan(acc / c.p as f64)
}

/// Yang-Zhang k = 0.34 / (1.34 + (n + 1) / (n - 1)), the value minimising the variance of the
/// combined estimator (Yang, Zhang 2000, with alpha = 1.34).
fn yz_k(n: usize) -> f64 {
    let n = n as f64;
    0.34 / (1.34 + (n + 1.0) / (n - 1.0))
}

/// Bar `i` of the window has open/high/low at `i` and close at `i + 1`; `close[0]` is the close
/// before the window.
fn f_vol_yang_zhang(v: &[&[f64]; 4], c: &Cfg, s: &mut Scratch) -> f64 {
    let (o, h, l, cl) = (v[0], v[1], v[2], v[3]);
    if !all_positive(&[o, h, l, cl]) {
        return f64::NAN;
    }
    let n = c.p;
    let (ov, oc) = (&mut s.a[..n], &mut s.b[..n]);
    let mut rs = 0.0;
    for i in 0..n {
        ov[i] = (o[i] / cl[i]).ln();
        oc[i] = (cl[i + 1] / o[i]).ln();
        rs += rs_term(o[i], h[i], l[i], cl[i + 1]);
    }
    let k = yz_k(n);
    sqrt_or_nan(sample_var(ov) + k * sample_var(oc) + (1.0 - k) * rs / n as f64)
}

// ---------------------------------------------------------------------------------------------
// Regression and co-movement. Inputs are (y, x).

fn f_beta(v: &[&[f64]; 2], _c: &Cfg, _s: &mut Scratch) -> f64 {
    fit(v[0], v[1]).map_or(f64::NAN, |f| f.beta)
}

fn f_alpha(v: &[&[f64]; 2], _c: &Cfg, _s: &mut Scratch) -> f64 {
    fit(v[0], v[1]).map_or(f64::NAN, |f| f.alpha)
}

fn f_r2(v: &[&[f64]; 2], _c: &Cfg, _s: &mut Scratch) -> f64 {
    fit(v[0], v[1]).map_or(f64::NAN, |f| f.r2)
}

fn f_residual(v: &[&[f64]; 2], c: &Cfg, _s: &mut Scratch) -> f64 {
    match fit(v[0], v[1]) {
        Some(f) => v[0][c.p - 1] - (f.alpha + f.beta * v[1][c.p - 1]),
        None => f64::NAN,
    }
}

fn f_spearman(v: &[&[f64]; 2], c: &Cfg, s: &mut Scratch) -> f64 {
    let Scratch { a, b, idx } = s;
    let n = c.p;
    ranks(v[0], idx, &mut a[..n]);
    let (ra, rb) = (&a[..n], &mut b[..n]);
    ranks(v[1], idx, rb);
    pearson(ra, rb)
}

// ---------------------------------------------------------------------------------------------
// Mean reversion and memory.

/// AR(1) fit x[t] = a + phi x[t-1] over the window; half-life = -ln 2 / ln phi for 0 < phi < 1.
fn f_half_life(v: &[&[f64]; 1], _c: &Cfg, _s: &mut Scratch) -> f64 {
    let x = v[0];
    let n = x.len();
    match fit(&x[1..], &x[..n - 1]) {
        Some(f) if f.beta > 0.0 && f.beta < 1.0 => -LN_2 / f.beta.ln(),
        _ => f64::NAN,
    }
}

/// Fixed-width-window fractional differencing: sum_k w[k] x[t-k], weights in `cfg.w`.
fn f_fracdiff(v: &[&[f64]; 1], c: &Cfg, _s: &mut Scratch) -> f64 {
    let x = v[0];
    let l = c.w.len();
    let mut acc = 0.0;
    for (k, w) in c.w.iter().enumerate() {
        acc += w * x[l - 1 - k];
    }
    acc
}

/// w[0] = 1, w[k] = -w[k-1] (d - k + 1) / k, cut at the first |w[k]| below `thr`.
fn ffd_weights(d: f64, thr: f64) -> Result<Vec<f64>, String> {
    let mut w = vec![1.0];
    let mut k = 1usize;
    loop {
        let next = -w[k - 1] * (d - k as f64 + 1.0) / k as f64;
        if next.abs() < thr {
            return Ok(w);
        }
        if w.len() >= MAX_FFD {
            return Err(format!("fracdiff({d}, {thr}) needs more than {MAX_FFD} weights"));
        }
        w.push(next);
        k += 1;
    }
}

/// Rescaled range of one block: range of the cumulative mean-deviations over the population
/// standard deviation. `None` for a flat block.
fn rescaled_range(x: &[f64]) -> Option<f64> {
    let n = x.len() as f64;
    let m = sum(x) / n;
    let (mut y, mut hi, mut lo, mut ss, mut raw) = (0.0, f64::MIN, f64::MAX, 0.0, 0.0);
    for &v in x {
        let d = v - m;
        y += d;
        hi = hi.max(y);
        lo = lo.min(y);
        ss += d * d;
        raw += v * v;
    }
    if degenerate(ss, raw) {
        return None;
    }
    Some((hi - lo) / (ss / n).sqrt())
}

/// Two-scale rescaled-range slope: H = ln(RS(p) / mean(RS of the two halves)) / ln(p / h).
/// One pass over the window and one over each half, so 2p operations per bar.
fn f_hurst(v: &[&[f64]; 1], _c: &Cfg, _s: &mut Scratch) -> f64 {
    let x = v[0];
    let p = x.len();
    let h = p / 2;
    match (rescaled_range(x), rescaled_range(&x[..h]), rescaled_range(&x[p - h..])) {
        (Some(full), Some(a), Some(b)) => {
            (full / ((a + b) / 2.0)).ln() / (p as f64 / h as f64).ln()
        }
        _ => f64::NAN,
    }
}

fn f_autocorr(v: &[&[f64]; 1], c: &Cfg, _s: &mut Scratch) -> f64 {
    let x = v[0];
    let (m, ss, raw) = spread(x);
    if degenerate(ss, raw) {
        return f64::NAN;
    }
    let lag = c.b as usize;
    let num: f64 = (lag..x.len()).map(|i| (x[i] - m) * (x[i - lag] - m)).sum();
    num / ss
}

// ---------------------------------------------------------------------------------------------
// Tails and risk. `cfg.a` carries q or mar.

/// Sorts a copy of the window into `buf`; returns the interpolated quantile at position
/// q (n - 1) (Hyndman-Fan type 7) and the index at or below that position.
fn sorted_quantile(v: &[f64], q: f64, buf: &mut [f64]) -> (f64, usize) {
    buf.copy_from_slice(v);
    buf.sort_unstable_by(f64::total_cmp);
    let pos = q * (v.len() - 1) as f64;
    let lo = (pos.floor() as usize).min(v.len() - 1);
    let at = if lo + 1 < v.len() {
        buf[lo] + (pos - lo as f64) * (buf[lo + 1] - buf[lo])
    } else {
        buf[lo]
    };
    (at, lo)
}

fn f_var_hist(v: &[&[f64]; 1], c: &Cfg, s: &mut Scratch) -> f64 {
    -sorted_quantile(v[0], c.a, &mut s.a[..c.p]).0
}

fn f_cvar_hist(v: &[&[f64]; 1], c: &Cfg, s: &mut Scratch) -> f64 {
    let (_, lo) = sorted_quantile(v[0], c.a, &mut s.a[..c.p]);
    -(s.a[..=lo].iter().sum::<f64>() / (lo + 1) as f64)
}

fn f_max_drawdown(v: &[&[f64]; 1], _c: &Cfg, _s: &mut Scratch) -> f64 {
    let x = v[0];
    if !all_positive(&[x]) {
        return f64::NAN;
    }
    let (mut peak, mut dd) = (x[0], 0.0f64);
    for &e in x {
        peak = peak.max(e);
        dd = dd.max((peak - e) / peak);
    }
    dd
}

fn f_downside_dev(v: &[&[f64]; 1], c: &Cfg, _s: &mut Scratch) -> f64 {
    let acc: f64 = v[0].iter().map(|&e| (e - c.a).min(0.0).powi(2)).sum();
    (acc / c.p as f64).sqrt()
}

fn f_sharpe(v: &[&[f64]; 1], _c: &Cfg, _s: &mut Scratch) -> f64 {
    let x = v[0];
    let (m, ss, raw) = spread(x);
    if degenerate(ss, raw) {
        return f64::NAN;
    }
    m / (ss / (x.len() - 1) as f64).sqrt()
}

// ---------------------------------------------------------------------------------------------
// Price structure.

/// Inputs (high, low, close); close window is (previous, current).
fn f_true_range_pct(v: &[&[f64]; 3], _c: &Cfg, _s: &mut Scratch) -> f64 {
    let (h, l, pc) = (v[0][0], v[1][0], v[2][0]);
    if !(pc > 0.0 && h > 0.0 && l > 0.0 && v[2][1] > 0.0) {
        return f64::NAN;
    }
    (h - l).max((h - pc).abs()).max((l - pc).abs()) / pc
}

/// Inputs (open, close); close window is (previous, current).
fn f_gap(v: &[&[f64]; 2], _c: &Cfg, _s: &mut Scratch) -> f64 {
    let (o, pc) = (v[0][0], v[1][0]);
    if !(o > 0.0 && pc > 0.0 && v[1][1] > 0.0) {
        return f64::NAN;
    }
    o / pc - 1.0
}

fn f_body_pct(v: &[&[f64]; 2], _c: &Cfg, _s: &mut Scratch) -> f64 {
    let (o, c) = (v[0][0], v[1][0]);
    if !(o > 0.0 && c > 0.0) {
        return f64::NAN;
    }
    (c - o) / o
}

/// (close - lowest low) / (highest high - lowest low) over the last `p` bars, by monotonic
/// deques: amortised O(1) per bar.
struct RangePos {
    hi: RollExt,
    lo: RollExt,
}

impl Kernel<3, 1> for RangePos {
    fn heap_bytes(&self) -> usize {
        self.hi.heap_bytes() + self.lo.heap_bytes()
    }

    fn lookback(&self) -> usize {
        0
    }

    #[inline(always)]
    fn step<H: Hist<3>>(&mut self, x: [f64; 3], _h: &H) -> [f64; 1] {
        let hh = self.hi.push(x[0]);
        let ll = self.lo.push(x[1]);
        let range = hh - ll;
        if !x[2].is_finite() || degenerate(range, hh.abs().max(ll.abs())) {
            return [f64::NAN];
        }
        [(x[2] - ll) / range]
    }
}

// ---------------------------------------------------------------------------------------------
// Builders.

type Made = Result<Box<dyn Node>, String>;

fn make_vol_close(p: &[ParamVal], i: &[usize], o: &[usize]) -> Made {
    let n = period(p, 2)?;
    mk([n + 1], Cfg::new(n), f_vol_close, i, o)
}

fn make_vol_parkinson(p: &[ParamVal], i: &[usize], o: &[usize]) -> Made {
    let n = period(p, 1)?;
    mk([n, n], Cfg::new(n), f_vol_parkinson, i, o)
}

fn make_vol_garman_klass(p: &[ParamVal], i: &[usize], o: &[usize]) -> Made {
    let n = period(p, 1)?;
    mk([n; 4], Cfg::new(n), f_vol_garman_klass, i, o)
}

fn make_vol_rogers_satchell(p: &[ParamVal], i: &[usize], o: &[usize]) -> Made {
    let n = period(p, 1)?;
    mk([n; 4], Cfg::new(n), f_vol_rogers_satchell, i, o)
}

fn make_vol_yang_zhang(p: &[ParamVal], i: &[usize], o: &[usize]) -> Made {
    let n = period(p, 2)?;
    mk([n, n, n, n + 1], Cfg::new(n), f_vol_yang_zhang, i, o)
}

fn make_beta(p: &[ParamVal], i: &[usize], o: &[usize]) -> Made {
    let n = period(p, 2)?;
    mk([n, n], Cfg::new(n), f_beta, i, o)
}

fn make_alpha(p: &[ParamVal], i: &[usize], o: &[usize]) -> Made {
    let n = period(p, 2)?;
    mk([n, n], Cfg::new(n), f_alpha, i, o)
}

fn make_r2(p: &[ParamVal], i: &[usize], o: &[usize]) -> Made {
    let n = period(p, 3)?;
    mk([n, n], Cfg::new(n), f_r2, i, o)
}

fn make_residual(p: &[ParamVal], i: &[usize], o: &[usize]) -> Made {
    let n = period(p, 3)?;
    mk([n, n], Cfg::new(n), f_residual, i, o)
}

fn make_spearman(p: &[ParamVal], i: &[usize], o: &[usize]) -> Made {
    let n = period(p, 3)?;
    mk([n, n], Cfg::new(n), f_spearman, i, o)
}

fn make_half_life(p: &[ParamVal], i: &[usize], o: &[usize]) -> Made {
    let n = period(p, 3)?;
    mk([n], Cfg::new(n), f_half_life, i, o)
}

fn make_fracdiff(p: &[ParamVal], i: &[usize], o: &[usize]) -> Made {
    let w = ffd_weights(float_param(p, 0)?, float_param(p, 1)?)?;
    let l = w.len();
    mk([l], Cfg { p: l, a: 0.0, b: 0.0, w }, f_fracdiff, i, o)
}

fn make_hurst(p: &[ParamVal], i: &[usize], o: &[usize]) -> Made {
    let n = period(p, 8)?;
    mk([n], Cfg::new(n), f_hurst, i, o)
}

fn make_autocorr(p: &[ParamVal], i: &[usize], o: &[usize]) -> Made {
    let n = period(p, 3)?;
    let lag = int_param(p, 1)?;
    if lag < 1 || lag + 2 > n {
        return Err(format!("lag must be in 1..=period-2, got lag {lag} for period {n}"));
    }
    mk([n], Cfg { p: n, a: 0.0, b: lag as f64, w: Vec::new() }, f_autocorr, i, o)
}

fn tail(p: &[ParamVal], f: WinFn<1>, i: &[usize], o: &[usize]) -> Made {
    let n = period(p, 2)?;
    let q = float_param(p, 1)?;
    if !(0.0..=1.0).contains(&q) {
        return Err(format!("q must be in [0, 1], got {q}"));
    }
    mk([n], Cfg { p: n, a: q, b: 0.0, w: Vec::new() }, f, i, o)
}

fn make_var_hist(p: &[ParamVal], i: &[usize], o: &[usize]) -> Made {
    tail(p, f_var_hist, i, o)
}

fn make_cvar_hist(p: &[ParamVal], i: &[usize], o: &[usize]) -> Made {
    tail(p, f_cvar_hist, i, o)
}

fn make_max_drawdown(p: &[ParamVal], i: &[usize], o: &[usize]) -> Made {
    let n = period(p, 2)?;
    mk([n], Cfg::new(n), f_max_drawdown, i, o)
}

fn make_downside_dev(p: &[ParamVal], i: &[usize], o: &[usize]) -> Made {
    let n = period(p, 1)?;
    let mar = float_param(p, 1)?;
    mk([n], Cfg { p: n, a: mar, b: 0.0, w: Vec::new() }, f_downside_dev, i, o)
}

fn make_sharpe(p: &[ParamVal], i: &[usize], o: &[usize]) -> Made {
    let n = period(p, 2)?;
    mk([n], Cfg::new(n), f_sharpe, i, o)
}

fn make_true_range_pct(_: &[ParamVal], i: &[usize], o: &[usize]) -> Made {
    mk([1, 1, 2], Cfg::new(1), f_true_range_pct, i, o)
}

fn make_gap(_: &[ParamVal], i: &[usize], o: &[usize]) -> Made {
    mk([1, 2], Cfg::new(1), f_gap, i, o)
}

fn make_body_pct(_: &[ParamVal], i: &[usize], o: &[usize]) -> Made {
    mk([1, 1], Cfg::new(1), f_body_pct, i, o)
}

fn make_range_pos(p: &[ParamVal], i: &[usize], o: &[usize]) -> Made {
    let n = period(p, 1)?;
    let k = RangePos { hi: RollExt::new(n, true), lo: RollExt::new(n, false) };
    Ok(Box::new(Adapter::new(k, regs::<3>(i, "inputs")?, regs::<1>(o, "outputs")?)))
}

// ---------------------------------------------------------------------------------------------
// Registry.

const fn field(name: &'static str) -> InputDef {
    InputDef { name, default_field: Some(name) }
}

const fn named(name: &'static str) -> InputDef {
    InputDef { name, default_field: None }
}

const IN_X: &[InputDef] = &[InputDef { name: "x", default_field: Some("close") }];
const IN_CLOSE: &[InputDef] = &[field("close")];
const IN_HL: &[InputDef] = &[field("high"), field("low")];
const IN_OHLC: &[InputDef] = &[field("open"), field("high"), field("low"), field("close")];
const IN_HLC: &[InputDef] = &[field("high"), field("low"), field("close")];
const IN_OC: &[InputDef] = &[field("open"), field("close")];
const IN_YX: &[InputDef] = &[named("y"), named("x")];
const IN_AB: &[InputDef] = &[named("a"), named("b")];

const fn period_param(min: i64) -> ParamDef {
    ParamDef { name: "period", kind: ParamKind::Int { min, max: MAX_PERIOD }, default: None }
}

const P1: &[ParamDef] = &[period_param(1)];
const P2: &[ParamDef] = &[period_param(2)];
const P3: &[ParamDef] = &[period_param(3)];
const P8: &[ParamDef] = &[period_param(8)];
const NONE: &[ParamDef] = &[];
const P_AUTOCORR: &[ParamDef] = &[
    period_param(3),
    ParamDef {
        name: "lag",
        kind: ParamKind::Int { min: 1, max: MAX_PERIOD },
        default: Some(ParamVal::Int(1)),
    },
];
const P_TAIL: &[ParamDef] = &[
    period_param(2),
    ParamDef { name: "q", kind: ParamKind::Float { min: 0.0, max: 1.0 }, default: None },
];
const P_DOWNSIDE: &[ParamDef] = &[
    period_param(1),
    ParamDef {
        name: "mar",
        kind: ParamKind::Float { min: -1e12, max: 1e12 },
        default: Some(ParamVal::Float(0.0)),
    },
];
const P_FRACDIFF: &[ParamDef] = &[
    ParamDef { name: "d", kind: ParamKind::Float { min: 0.0, max: 2.0 }, default: None },
    ParamDef { name: "threshold", kind: ParamKind::Float { min: 1e-12, max: 1.0 }, default: None },
];

fn warm_p_minus_1(p: &[ParamVal]) -> usize {
    int_param(p, 0).map_or(0, |v| v.saturating_sub(1))
}

fn warm_p(p: &[ParamVal]) -> usize {
    int_param(p, 0).unwrap_or(0)
}

fn warm_one(_: &[ParamVal]) -> usize {
    1
}

fn warm_zero(_: &[ParamVal]) -> usize {
    0
}

fn warm_fracdiff(p: &[ParamVal]) -> usize {
    match (float_param(p, 0), float_param(p, 1)) {
        (Ok(d), Ok(t)) => ffd_weights(d, t).map_or(0, |w| w.len() - 1),
        _ => 0,
    }
}

macro_rules! def {
    ($name:literal, $inputs:expr, $params:expr, $warm:expr, $doc:expr, $refs:expr, $inv:expr,
     $nan:expr, $domain:expr, $make:expr) => {
        FnDef {
            name: $name,
            family: Family::Quant,
            inputs: $inputs,
            params: $params,
            outputs: &["value"],
            out_kind: OutKind::Series,
            warmup: $warm,
            memory: Memory::Finite,
            legacy: None,
            talib_name: None,
            doc: $doc,
            refs: $refs,
            rev: 1,
            invariance: $inv,
            causal: true,
            nan_rule: $nan,
            domain: $domain,
            reference: Reference::Hand(concat!("hand_", $name)),
            relation: Relation::NoLibrary,
            make: $make,
        }
    };
}

use Domain::{Any, Positive};
use Invariance::{ScaleEquivariant, ScaleInvariant, ShiftInvariant};
use NanRule::{Elementwise, Lagged, Windowed};

const REF_HULL: &str = "Hull, J. (2018) Options, Futures, and Other Derivatives, 10th ed., Pearson, on estimating volatility from historical data";
const REF_PARKINSON: &str = "Parkinson, M. (1980) The Extreme Value Method for Estimating the Variance of the Rate of Return, Journal of Business 53(1), 61-65";
const REF_GK: &str = "Garman, M.B. and Klass, M.J. (1980) On the Estimation of Security Price Volatilities from Historical Data, Journal of Business 53(1), 67-78";
const REF_RS: &str = "Rogers, L.C.G. and Satchell, S.E. (1991) Estimating Variance From High, Low and Closing Prices, Annals of Applied Probability 1(4), 504-512";
const REF_YZ: &str = "Yang, D. and Zhang, Q. (2000) Drift-Independent Volatility Estimation Based on High, Low, Open, and Close Prices, Journal of Business 73(3), 477-492";
const REF_SHARPE64: &str = "Sharpe, W.F. (1964) Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk, Journal of Finance 19(3), 425-442";
const REF_JENSEN: &str = "Jensen, M.C. (1968) The Performance of Mutual Funds in the Period 1945-1964, Journal of Finance 23(2), 389-416";
const REF_GREENE: &str =
    "Greene, W.H. (2012) Econometric Analysis, 7th ed., Pearson, chapter 3 (least squares)";
const REF_SPEARMAN: &str = "Spearman, C. (1904) The Proof and Measurement of Association between Two Things, American Journal of Psychology 15(1), 72-101";
const REF_NISON: &str =
    "Nison, S. (1991) Japanese Candlestick Charting Techniques, New York Institute of Finance";

pub const DEFS: &[FnDef] = &[
    def!(
        "vol_close",
        IN_CLOSE,
        P2,
        warm_p,
        "Close-to-close realised volatility: the sample standard deviation (divisor p - 1) of the \
         `period` log returns ln(c[t]/c[t-1]) ending at the current bar, so the window spans \
         `period + 1` closes and the first value is at bar `period`. Per-bar volatility; \
         annualising (times sqrt of bars per year) is the caller's. A non-positive close in the \
         window gives NaN. O(p) per bar.",
        &[REF_HULL],
        &[ScaleInvariant],
        Windowed,
        Positive,
        make_vol_close
    ),
    def!(
        "vol_parkinson",
        IN_HL,
        P1,
        warm_p_minus_1,
        "Parkinson (1980) range volatility: sqrt(mean over the window of ln(high/low)^2 / (4 \
         ln 2)). Per-bar volatility; annualising is the caller's. Assumes driftless continuous \
         prices, so it understates when the market gaps. A non-positive price in the window gives \
         NaN. O(p) per bar.",
        &[REF_PARKINSON],
        &[ScaleInvariant],
        Windowed,
        Positive,
        make_vol_parkinson
    ),
    def!(
        "vol_garman_klass",
        IN_OHLC,
        P1,
        warm_p_minus_1,
        "Garman-Klass (1980) volatility: sqrt(mean of 0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2). \
         Per-bar volatility; annualising is the caller's. Assumes no drift and no overnight gap. \
         A non-positive price in the window gives NaN, as does a negative mean (inconsistent \
         OHLC). O(p) per bar.",
        &[REF_GK],
        &[ScaleInvariant],
        Windowed,
        Positive,
        make_vol_garman_klass
    ),
    def!(
        "vol_rogers_satchell",
        IN_OHLC,
        P1,
        warm_p_minus_1,
        "Rogers-Satchell (1991) volatility: sqrt(mean of ln(H/C) ln(H/O) + ln(L/C) ln(L/O)); \
         unbiased with drift. Per-bar volatility; annualising is the caller's. Assumes no \
         overnight gap. A non-positive price in the window gives NaN, as does a negative mean \
         (inconsistent OHLC). O(p) per bar.",
        &[REF_RS],
        &[ScaleInvariant],
        Windowed,
        Positive,
        make_vol_rogers_satchell
    ),
    def!(
        "vol_yang_zhang",
        IN_OHLC,
        P2,
        warm_p,
        "Yang-Zhang (2000) volatility over `period` bars: sqrt(s_o^2 + k s_c^2 + (1 - k) \
         s_rs^2), with s_o^2 the sample variance (divisor n - 1) of the overnight log return \
         ln(O[t]/C[t-1]), s_c^2 that of ln(C[t]/O[t]), s_rs^2 the mean Rogers-Satchell term, and \
         k = 0.34 / (1.34 + (n + 1) / (n - 1)) as in the paper. Handles drift and overnight gaps. \
         The window needs the close before its first bar, so the first value is at bar `period`. \
         Per-bar volatility; annualising is the caller's. A non-positive price in the window \
         gives NaN. O(p) per bar.",
        &[REF_YZ],
        &[ScaleInvariant],
        Windowed,
        Positive,
        make_vol_yang_zhang
    ),
    def!(
        "rolling_beta",
        IN_YX,
        P2,
        warm_p_minus_1,
        "OLS slope of `y` on `x` over the window, on the inputs as given (levels or returns, \
         whichever you pass). Differs from TA-Lib BETA, which regresses returns derived from its \
         inputs. NaN when `x` is flat or nearly so: the centred sum of squares of x is at most \
         1e-12 of its raw sum of squares. O(p) per bar.",
        &[REF_SHARPE64, REF_GREENE],
        &[ScaleInvariant, ShiftInvariant],
        Windowed,
        Any,
        make_beta
    ),
    def!(
        "rolling_alpha",
        IN_YX,
        P2,
        warm_p_minus_1,
        "OLS intercept of `y` on `x` over the window (mean y minus beta times mean x), on the \
         inputs as given. NaN when `x` is flat or nearly so (see rolling_beta). Not annualised. \
         O(p) per bar.",
        &[REF_JENSEN, REF_GREENE],
        &[ScaleEquivariant],
        Windowed,
        Any,
        make_alpha
    ),
    def!(
        "rolling_r2",
        IN_YX,
        P3,
        warm_p_minus_1,
        "R-squared of the OLS fit of `y` on `x` over the window: squared Pearson correlation. \
         NaN when either input is flat or nearly so. O(p) per bar.",
        &[REF_GREENE],
        &[ScaleInvariant, ShiftInvariant],
        Windowed,
        Any,
        make_r2
    ),
    def!(
        "rolling_residual",
        IN_YX,
        P3,
        warm_p_minus_1,
        "Residual of the current bar against the OLS line fitted on the window that includes the \
         current bar: y[t] - (alpha + beta x[t]). In-sample, so it shrinks toward zero for short \
         windows. NaN when `x` is flat or nearly so. O(p) per bar.",
        &[REF_GREENE],
        &[ScaleEquivariant, ShiftInvariant],
        Windowed,
        Any,
        make_residual
    ),
    def!(
        "rolling_corr_spearman",
        IN_AB,
        P3,
        warm_p_minus_1,
        "Spearman rank correlation of `a` and `b` over the window: Pearson on average ranks, so \
         ties share their mean rank. NaN when either window is constant. O(p log p) per bar \
         (a sort per input).",
        &[REF_SPEARMAN],
        &[ScaleInvariant],
        Windowed,
        Any,
        make_spearman
    ),
    def!(
        "half_life",
        IN_X,
        P3,
        warm_p_minus_1,
        "Ornstein-Uhlenbeck half-life in bars from an AR(1) fit x[t] = a + phi x[t-1] over the \
         window: -ln 2 / ln phi. NaN unless 0 < phi < 1, so a trending, random-walk or \
         oscillating window gives NaN, never a negative. NaN when the lagged series is flat. \
         O(p) per bar.",
        &[
            "Uhlenbeck, G.E. and Ornstein, L.S. (1930) On the Theory of the Brownian Motion, Physical Review 36, 823-841",
            "Chan, E.P. (2013) Algorithmic Trading: Winning Strategies and Their Rationale, Wiley, chapter 2",
        ],
        &[ScaleInvariant, ShiftInvariant],
        Windowed,
        Any,
        make_half_life
    ),
    def!(
        "fracdiff",
        IN_X,
        P_FRACDIFF,
        warm_fracdiff,
        "Fixed-width-window fractional differencing of order `d`: sum of w[k] x[t-k] with w[0] = \
         1, w[k] = -w[k-1] (d - k + 1) / k, the series of weights cut at the first |w[k]| below \
         `threshold`. Causal: the window is the weights' length L, so the first value is at bar \
         L - 1; a smaller threshold keeps more memory and a longer window (refused above 2^20). \
         d = 1 is the first difference. O(L) per bar.",
        &[
            "Lopez de Prado, M. (2018) Advances in Financial Machine Learning, Wiley, chapter 5",
            "Hosking, J.R.M. (1981) Fractional differencing, Biometrika 68(1), 165-176",
        ],
        &[ScaleEquivariant],
        Windowed,
        Any,
        make_fracdiff
    ),
    def!(
        "hurst_rs",
        IN_X,
        P8,
        warm_p_minus_1,
        "Rescaled-range Hurst exponent over the window, as the slope between two scales: H = \
         ln(RS(p) / mean(RS of the first and last floor(p/2) values)) / ln(p / floor(p/2)), RS = \
         range of the cumulative mean-deviations over the population standard deviation. Feed it \
         increments (returns), not levels. Two scales only and not bias-corrected for short \
         samples (Anis-Lloyd), so read it relative to its own history. NaN when any block is \
         flat. Three passes over at most p values per bar: O(p).",
        &["Hurst, H.E. (1951) Long-term storage capacity of reservoirs, Transactions of the American Society of Civil Engineers 116, 770-808"],
        &[ScaleInvariant, ShiftInvariant],
        Windowed,
        Any,
        make_hurst
    ),
    def!(
        "autocorr",
        IN_X,
        P_AUTOCORR,
        warm_p_minus_1,
        "Sample autocorrelation at `lag` over the window: sum over i >= lag of (x[i] - m)(x[i-lag] \
         - m) divided by the sum of (x[i] - m)^2, m the window mean. `lag` defaults to 1 and must not exceed period - 2. \
         NaN when the window is flat or nearly so. O(p) per bar.",
        &["Box, G.E.P. and Jenkins, G.M. (1976) Time Series Analysis: Forecasting and Control, Holden-Day"],
        &[ScaleInvariant, ShiftInvariant],
        Windowed,
        Any,
        make_autocorr
    ),
    def!(
        "rolling_var_hist",
        IN_X,
        P_TAIL,
        warm_p_minus_1,
        "Historical value-at-risk of the returns `x` over the window at tail probability `q` \
         (0.05 for the 5% tail): minus the q-quantile with linear interpolation at position \
         q(p - 1) of the sorted window (Hyndman-Fan type 7). A loss is positive. O(p log p) per \
         bar (a sort).",
        &[
            "Hyndman, R.J. and Fan, Y. (1996) Sample quantiles in statistical packages, The American Statistician 50(4), 361-365",
            "Jorion, P. (2006) Value at Risk, 3rd ed., McGraw-Hill",
        ],
        &[ScaleEquivariant],
        Windowed,
        Any,
        make_var_hist
    ),
    def!(
        "rolling_cvar_hist",
        IN_X,
        P_TAIL,
        warm_p_minus_1,
        "Historical expected shortfall of the returns `x` over the window at tail probability \
         `q`: minus the mean of the sorted observations up to and including the index \
         floor(q(p - 1)). A loss is positive and never smaller than rolling_var_hist at the \
         same q. O(p log p) per bar (a sort).",
        &["Rockafellar, R.T. and Uryasev, S. (2000) Optimization of conditional value-at-risk, Journal of Risk 2(3), 21-41"],
        &[ScaleEquivariant],
        Windowed,
        Any,
        make_cvar_hist
    ),
    def!(
        "max_drawdown",
        IN_X,
        P2,
        warm_p_minus_1,
        "Largest peak-to-trough fall of the level series `x` inside the window, as a fraction of \
         the peak (0.25 is a 25% drawdown); 0 for a rising window. A non-positive level in the \
         window gives NaN. O(p) per bar.",
        &["Magdon-Ismail, M. and Atiya, A.F. (2004) Maximum drawdown, Risk 17(10), 99-102"],
        &[ScaleInvariant],
        Windowed,
        Positive,
        make_max_drawdown
    ),
    def!(
        "downside_dev",
        IN_X,
        P_DOWNSIDE,
        warm_p_minus_1,
        "Downside deviation of the returns `x` below the minimum acceptable return `mar` over \
         the window: sqrt(mean of min(x - mar, 0)^2), divisor p. Per-bar; annualising is the \
         caller's. O(p) per bar.",
        &["Sortino, F.A. and van der Meer, R. (1991) Downside risk, Journal of Portfolio Management 17(4), 27-31"],
        &[],
        Windowed,
        Any,
        make_downside_dev
    ),
    def!(
        "rolling_sharpe",
        IN_X,
        P2,
        warm_p_minus_1,
        "Mean over sample standard deviation (divisor p - 1) of the returns `x` in the window. \
         No risk-free rate and no annualisation: both are the caller's. NaN when the window is \
         flat or nearly so. O(p) per bar.",
        &["Sharpe, W.F. (1966) Mutual fund performance, Journal of Business 39(1), 119-138"],
        &[ScaleInvariant],
        Windowed,
        Any,
        make_sharpe
    ),
    def!(
        "true_range_pct",
        IN_HLC,
        NONE,
        warm_one,
        "True range as a fraction of the previous close: max(H - L, |H - Cprev|, |L - Cprev|) / \
         Cprev. A non-positive price gives NaN. O(1) per bar.",
        &["Wilder, J.W. (1978) New Concepts in Technical Trading Systems, Trend Research"],
        &[ScaleInvariant],
        Lagged,
        Positive,
        make_true_range_pct
    ),
    def!(
        "gap",
        IN_OC,
        NONE,
        warm_one,
        "Overnight gap: today's open over the previous close, minus 1. Today's close must be \
         present (a bar is complete) but is not used. A non-positive price gives NaN. O(1) per \
         bar.",
        &[REF_NISON],
        &[ScaleInvariant],
        Lagged,
        Positive,
        make_gap
    ),
    def!(
        "body_pct",
        IN_OC,
        NONE,
        warm_zero,
        "Candle body as a signed fraction of the open: (close - open) / open. A non-positive \
         price gives NaN. O(1) per bar.",
        &[REF_NISON],
        &[ScaleInvariant],
        Elementwise,
        Positive,
        make_body_pct
    ),
    def!(
        "range_pos",
        IN_HLC,
        P1,
        warm_p_minus_1,
        "Where the close sits in the rolling range: (close - lowest low) / (highest high - \
         lowest low) over `period` bars, 0 at the low and 1 at the high. NaN when the range is \
         flat (at most 1e-12 of the price level). Amortised O(1) per bar.",
        &["Williams, L. (1973) How I Made $1,000,000 Last Year Trading Commodities, Windsor Books"],
        &[ScaleInvariant, ShiftInvariant],
        Windowed,
        Any,
        make_range_pos
    ),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::kernel::{canon_bits, Regs};
    use crate::quant::registry::lookup;

    fn int(v: i64) -> ParamVal {
        ParamVal::Int(v)
    }

    fn fl(v: f64) -> ParamVal {
        ParamVal::Float(v)
    }

    fn run(name: &str, params: &[ParamVal], cols: &[&[f64]]) -> Vec<f64> {
        let ins: Vec<usize> = (0..cols.len()).collect();
        let mut node = (lookup(name).unwrap().make)(params, &ins, &[cols.len()]).unwrap();
        let mut out = vec![0.0; cols[0].len()];
        node.run_batch(cols, &mut [&mut out[..]]);
        out
    }

    fn stream(name: &str, params: &[ParamVal], cols: &[&[f64]]) -> Vec<f64> {
        let ins: Vec<usize> = (0..cols.len()).collect();
        let mut node = (lookup(name).unwrap().make)(params, &ins, &[cols.len()]).unwrap();
        let mut regs = Regs::new(cols.len() + 1);
        (0..cols[0].len())
            .map(|i| {
                for (j, c) in cols.iter().enumerate() {
                    regs.set(j, c[i]);
                }
                node.step(&mut regs);
                regs.get(cols.len())
            })
            .collect()
    }

    fn near(got: f64, want: f64, tol: f64) {
        assert!((got - want).abs() <= tol, "got {got} want {want}");
    }

    fn ex(v: &[f64]) -> Vec<f64> {
        v.iter().map(|a| a.exp()).collect()
    }

    // Splitmix64 with Box-Muller normals: seeded, no dependency.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        /// Uniform on (0, 1].
        fn unif(&mut self) -> f64 {
            ((self.next() >> 11) + 1) as f64 / (1u64 << 53) as f64
        }

        fn normal(&mut self) -> f64 {
            let (u1, u2) = (self.unif(), self.unif());
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        }
    }

    #[test]
    fn hand_vol_close() {
        // Closes 100, 110, 99: r = ln 1.1, ln 0.9. Two points, divisor 1:
        // var = (r1 - r2)^2 / 2, so sd = ln(1.1/0.9) / sqrt 2 = ln(11/9) / sqrt 2.
        let out = run("vol_close", &[int(2)], &[&[100.0, 110.0, 99.0]]);
        assert!(out[0].is_nan() && out[1].is_nan());
        near(out[2], (11.0f64 / 9.0).ln() / 2.0f64.sqrt(), 1e-14);
    }

    #[test]
    fn hand_vol_parkinson() {
        // Bars with ln(H/L) = 1 and 2: mean of squares 2.5, so sqrt(2.5 / (4 ln 2)).
        let (h, l) = (ex(&[1.0, 2.0]), ex(&[0.0, 0.0]));
        let out = run("vol_parkinson", &[int(2)], &[&h, &l]);
        assert!(out[0].is_nan());
        near(out[1], (2.5 / (4.0 * LN_2)).sqrt(), 1e-12);
    }

    #[test]
    fn hand_vol_garman_klass() {
        // O = 1, H = e, L = 1, C = e^0.5: ln(H/L) = 1, ln(C/O) = 0.5.
        // 0.5 * 1 - (2 ln 2 - 1) * 0.25.
        let (o, h, l, c) = (ex(&[0.0]), ex(&[1.0]), ex(&[0.0]), ex(&[0.5]));
        let out = run("vol_garman_klass", &[int(1)], &[&o, &h, &l, &c]);
        near(out[0], (0.5 - (2.0 * LN_2 - 1.0) * 0.25).sqrt(), 1e-12);
    }

    #[test]
    fn hand_vol_rogers_satchell() {
        // O = 1, H = e, L = e^-1, C = e^0.5:
        // ln(H/C) ln(H/O) = 0.5 * 1, ln(L/C) ln(L/O) = (-1.5)(-1) = 1.5, total 2.
        let (o, h, l, c) = (ex(&[0.0]), ex(&[1.0]), ex(&[-1.0]), ex(&[0.5]));
        let out = run("vol_rogers_satchell", &[int(1)], &[&o, &h, &l, &c]);
        near(out[0], 2.0f64.sqrt(), 1e-12);
    }

    #[test]
    fn hand_vol_yang_zhang() {
        // Log prices. Close before the window 0. Bar 1: O .02 H .05 L -.01 C .03.
        // Bar 2: O .01 H .04 L -.02 C 0.
        // Overnight: .02 - 0 = .02 and .01 - .03 = -.02, mean 0, variance (2 * 4e-4) / 1 = 8e-4.
        // Open-close: .01 and -.01, mean 0, variance 2e-4.
        // RS bar 1: (.05-.03)(.05-.02) + (-.01-.03)(-.01-.02) = 6e-4 + 12e-4 = 18e-4;
        // bar 2: (.04-0)(.04-.01) + (-.02-0)(-.02-.01) = 12e-4 + 6e-4 = 18e-4. Mean 18e-4.
        // n = 2: k = 0.34 / (1.34 + 3 / 1) = 0.34 / 4.34.
        // Index 0 holds only the earlier close; the open, high and low there are absent.
        let nan = f64::NAN;
        let o = [nan, 0.02f64.exp(), 0.01f64.exp()];
        let h = [nan, 0.05f64.exp(), 0.04f64.exp()];
        let l = [nan, (-0.01f64).exp(), (-0.02f64).exp()];
        let c = ex(&[0.0, 0.03, 0.0]);
        let out = run("vol_yang_zhang", &[int(2)], &[&o, &h, &l, &c]);
        let k = 0.34 / 4.34;
        let want = (8e-4 + k * 2e-4 + (1.0 - k) * 18e-4f64).sqrt();
        assert!(out[0].is_nan() && out[1].is_nan());
        near(out[2], want, 1e-12);
        // The paper's k at n = 2 and n = 11.
        near(yz_k(2), 0.34 / 4.34, 1e-15);
        near(yz_k(11), 0.34 / (1.34 + 12.0 / 10.0), 1e-15);
    }

    #[test]
    fn vol_nonpositive_price_is_nan() {
        let c = [100.0, 101.0, 0.0, 102.0, 103.0, 104.0];
        let out = run("vol_close", &[int(2)], &[&c]);
        // Windows of 3 closes holding the 0 are bars 2, 3, 4.
        assert!(out[2].is_nan() && out[3].is_nan() && out[4].is_nan());
        assert!(out[5].is_finite());
        let h = [2.0, 2.0, -1.0, 2.0];
        let l = [1.0, 1.0, 1.0, 1.0];
        let out = run("vol_parkinson", &[int(1)], &[&h, &l]);
        assert!(out[2].is_nan() && out[3].is_finite());
    }

    /// Driftless GBM with exact bar extremes: each of `m` substeps' maximum and minimum is drawn
    /// from the Brownian bridge law between its endpoints (P(max > x) = exp(-2 (x - a)(x - b) /
    /// s^2)), so the bar's high and low are exact up to the dependence of max and min inside one
    /// substep, an event of probability about 1/m that moves the range by one substep's scale.
    fn gbm(n: usize, m: usize, s_day: f64, s_night: f64, seed: u64) -> [Vec<f64>; 4] {
        let mut rng = Rng(seed);
        let s = s_day / (m as f64).sqrt();
        let (mut o, mut h, mut l, mut c) = (vec![], vec![], vec![], vec![]);
        let mut prev = 0.0;
        for _ in 0..n {
            let open = prev + s_night * rng.normal();
            let (mut a, mut hi, mut lo) = (0.0f64, 0.0f64, 0.0f64);
            for _ in 0..m {
                let b = a + s * rng.normal();
                let (u1, u2) = (rng.unif(), rng.unif());
                let d2 = (b - a) * (b - a);
                hi = hi.max(((a + b) + (d2 - 2.0 * s * s * u1.ln()).sqrt()) / 2.0);
                lo = lo.min(((a + b) - (d2 - 2.0 * s * s * u2.ln()).sqrt()) / 2.0);
                a = b;
            }
            o.push(open.exp());
            h.push((open + hi).exp());
            l.push((open + lo).exp());
            c.push((open + a).exp());
            prev = open + a;
        }
        [o, h, l, c]
    }

    #[test]
    fn estimators_are_unbiased_on_driftless_gbm() {
        // Per-bar day sd 0.02 and overnight sd 0.01. Close-to-close and Yang-Zhang see both:
        // variance 0.02^2 + 0.01^2; the intraday estimators see 0.02^2.
        let (s_day, s_night, p) = (0.02f64, 0.01f64, 25usize);
        let pp = [int(p as i64)];
        let [o, h, l, c] = gbm(30_000, 50, s_day, s_night, 0x5EED_2026);
        let day = s_day * s_day;
        let total = day + s_night * s_night;
        let cases: [(&str, Vec<f64>, f64); 5] = [
            ("vol_close", run("vol_close", &pp, &[&c]), total),
            ("vol_parkinson", run("vol_parkinson", &pp, &[&h, &l]), day),
            ("vol_garman_klass", run("vol_garman_klass", &pp, &[&o, &h, &l, &c]), day),
            ("vol_rogers_satchell", run("vol_rogers_satchell", &pp, &[&o, &h, &l, &c]), day),
            ("vol_yang_zhang", run("vol_yang_zhang", &pp, &[&o, &h, &l, &c]), total),
        ];
        for (name, out, want) in cases {
            // Non-overlapping windows are independent, so the standard error of the mean of
            // the variance estimates is sd / sqrt(K); the bound is five of them.
            let first = out.iter().position(|v| v.is_finite()).unwrap();
            let v: Vec<f64> = (first..out.len()).step_by(p).map(|i| out[i] * out[i]).collect();
            let k = v.len() as f64;
            let mean = v.iter().sum::<f64>() / k;
            let sd = (v.iter().map(|e| (e - mean) * (e - mean)).sum::<f64>() / (k - 1.0)).sqrt();
            let se = sd / k.sqrt();
            assert!(
                (mean - want).abs() <= 5.0 * se,
                "{name}: mean variance {mean:e} vs {want:e}, 5 se = {:e}",
                5.0 * se
            );
        }
    }

    #[test]
    fn hand_rolling_beta() {
        // y = 2x + 1: means 3 and 7, sxx = 10, sxy = 20.
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [3.0, 5.0, 7.0, 9.0, 11.0];
        let out = run("rolling_beta", &[int(5)], &[&y, &x]);
        assert_eq!(out[4], 2.0);
        assert!(out[3].is_nan());
    }

    #[test]
    fn hand_rolling_alpha() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [3.0, 5.0, 7.0, 9.0, 11.0];
        let out = run("rolling_alpha", &[int(5)], &[&y, &x]);
        assert_eq!(out[4], 1.0);
    }

    #[test]
    fn hand_rolling_r2() {
        // x = 1..4, y = 1,3,2,4: means 2.5, sxx = 5, syy = 5, sxy = 2.25 - .25 - .25 + 2.25 = 4,
        // so r2 = 16 / 25.
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [1.0, 3.0, 2.0, 4.0];
        near(run("rolling_r2", &[int(4)], &[&y, &x])[3], 0.64, 1e-15);
    }

    #[test]
    fn hand_rolling_residual() {
        // Same data: beta = 4/5 = .8, alpha = 2.5 - .8 * 2.5 = .5, residual at x = 4:
        // 4 - (.5 + 3.2) = .3.
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [1.0, 3.0, 2.0, 4.0];
        near(run("rolling_residual", &[int(4)], &[&y, &x])[3], 0.3, 1e-15);
        near(run("rolling_beta", &[int(4)], &[&y, &x])[3], 0.8, 1e-15);
        near(run("rolling_alpha", &[int(4)], &[&y, &x])[3], 0.5, 1e-15);
    }

    #[test]
    fn regression_degenerate_windows_are_nan() {
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // Flat x, and x that varies by 1e-13 around 1e8 (below the spacing of f64 there: flat).
        let flat = [7.0; 6];
        let near_flat: Vec<f64> = (0..6).map(|i| 1e8 + i as f64 * 1e-13).collect();
        // Nearly flat but representable: spread 1e-6 around 1e6, centred / raw ~ 1e-25.
        let tiny: Vec<f64> = (0..6).map(|i| 1e6 + i as f64 * 1e-6).collect();
        for x in [&flat[..], &near_flat[..], &tiny[..]] {
            for name in ["rolling_beta", "rolling_alpha", "rolling_residual", "rolling_r2"] {
                let out = run(name, &[int(5)], &[&y, x]);
                assert!(out[5].is_nan(), "{name}");
            }
        }
        // A flat y is a valid fit with slope 0 but r2 is undefined.
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let yf = [4.0; 5];
        assert_eq!(run("rolling_beta", &[int(5)], &[&yf, &x])[4], 0.0);
        assert!(run("rolling_r2", &[int(5)], &[&yf, &x])[4].is_nan());
        // Ordinary price-scale data is not degenerate: 100 + noise of 0.01, y = x / 2.
        let xp = [100.00, 100.01, 99.99, 100.02, 100.00];
        let yp = [50.0, 50.005, 49.995, 50.01, 50.0];
        near(run("rolling_beta", &[int(5)], &[&yp, &xp])[4], 0.5, 1e-9);
    }

    #[test]
    fn hand_rolling_corr_spearman() {
        // a = 1..5, b = 1,3,2,5,4 (ranks equal values). Mean 3; da -2..2, db -2,0,-1,2,1;
        // sxy = 4 + 0 + 0 + 2 + 2 = 8, sxx = syy = 10: 0.8.
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [1.0, 3.0, 2.0, 5.0, 4.0];
        near(run("rolling_corr_spearman", &[int(5)], &[&a, &b])[4], 0.8, 1e-15);
        // Ties: b = 1,1,2,2 gets ranks 1.5,1.5,3.5,3.5. da = -1.5,-.5,.5,1.5, db = -1,-1,1,1:
        // sxy = 4, sxx = 5, syy = 4: 4 / sqrt 20.
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0, 1.0, 2.0, 2.0];
        near(run("rolling_corr_spearman", &[int(4)], &[&a, &b])[3], 4.0 / 20.0f64.sqrt(), 1e-15);
        // A monotone but nonlinear relation is 1; a constant window is NaN.
        let c: Vec<f64> = a.iter().map(|v| v.powi(5)).collect();
        near(run("rolling_corr_spearman", &[int(4)], &[&a, &c])[3], 1.0, 1e-15);
        let k = [3.0; 4];
        assert!(run("rolling_corr_spearman", &[int(4)], &[&a, &k])[3].is_nan());
    }

    #[test]
    fn hand_half_life() {
        // 8, 4, 2, 1: the pairs (8,4), (4,2), (2,1) lie on x[t] = 0.5 x[t-1] exactly, so phi =
        // 0.5 and the half-life is -ln 2 / ln 0.5 = 1.
        let out = run("half_life", &[int(4)], &[&[8.0, 4.0, 2.0, 1.0]]);
        near(out[3], 1.0, 1e-15);
        // x[t] = 0.9 x[t-1] + 1 lies on a line with phi = 0.9: half-life -ln 2 / ln 0.9.
        let mut x = vec![20.0];
        for _ in 0..9 {
            x.push(0.9 * x.last().unwrap() + 1.0);
        }
        near(run("half_life", &[int(10)], &[&x])[9], -LN_2 / 0.9f64.ln(), 1e-9);
    }

    #[test]
    fn half_life_of_a_non_reverting_window_is_nan() {
        // Pairs (1,2), (2,3), (3,4): phi = 1 exactly. Growth (phi = 2), oscillation (phi =
        // -0.5) and a flat window are NaN too, never negative.
        assert!(run("half_life", &[int(4)], &[&[1.0, 2.0, 3.0, 4.0]])[3].is_nan());
        assert!(run("half_life", &[int(4)], &[&[1.0, 2.0, 4.0, 8.0]])[3].is_nan());
        assert!(run("half_life", &[int(4)], &[&[8.0, -4.0, 2.0, -1.0]])[3].is_nan());
        assert!(run("half_life", &[int(4)], &[&[5.0; 4]])[3].is_nan());
    }

    #[test]
    fn hand_fracdiff() {
        // d = 0.5: w1 = -0.5, w2 = -(-0.5)(0.5 - 2 + 1)/2 = 0.5 * (-0.5) / 2 = -0.125,
        // w3 = -(-0.125)(0.5 - 3 + 1)/3 = 0.125 * (-1.5) / 3 = -0.0625. With threshold 0.1
        // |w3| is cut, so the weights are [1, -0.5, -0.125].
        // x = 4, 6, 10: 10 - 0.5 * 6 - 0.125 * 4 = 6.5.
        let out = run("fracdiff", &[fl(0.5), fl(0.1)], &[&[4.0, 6.0, 10.0]]);
        assert!(out[0].is_nan() && out[1].is_nan());
        assert_eq!(out[2], 6.5);
        // d = 1 is the first difference: w2 = -(-1)(1 - 2 + 1)/2 = 0 is cut.
        let out = run("fracdiff", &[fl(1.0), fl(1e-6)], &[&[1.0, 4.0, 9.0]]);
        assert!(out[0].is_nan());
        assert_eq!((out[1], out[2]), (3.0, 5.0));
        // d = 0 is the identity.
        assert_eq!(run("fracdiff", &[fl(0.0), fl(1e-6)], &[&[2.0, 3.0]]), vec![2.0, 3.0]);
        // Warm-up matches the window of 3.
        let def = lookup("fracdiff").unwrap();
        assert_eq!((def.warmup)(&[fl(0.5), fl(0.1)]), 2);
        // A threshold that needs more than 2^20 weights is refused at build.
        assert!((def.make)(&[fl(0.01), fl(1e-12)], &[0], &[1]).is_err());
    }

    #[test]
    fn hand_hurst_rs() {
        // Window [0,2,0,2,0,0,2,2], mean 1, deviations -1,1,-1,1,-1,-1,1,1, cumulative
        // -1,0,-1,0,-1,-2,-1,0: range 2, population sd 1: RS = 2.
        // First half [0,2,0,2]: cumulative -1,0,-1,0: range 1, sd 1: RS 1.
        // Last half [0,0,2,2]: cumulative -1,-2,-1,0: range 2, sd 1: RS 2. Mean 1.5.
        // H = ln(2 / 1.5) / ln(8 / 4) = ln(4/3) / ln 2.
        let x = [0.0, 2.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0];
        let out = run("hurst_rs", &[int(8)], &[&x]);
        near(out[7], (4.0f64 / 3.0).ln() / LN_2, 1e-15);
        // A flat block gives NaN.
        assert!(run("hurst_rs", &[int(8)], &[&[1.0; 8]])[7].is_nan());
        // Scale and shift of the window leave it unchanged.
        let y: Vec<f64> = x.iter().map(|v| 3.0 * v + 10.0).collect();
        near(run("hurst_rs", &[int(8)], &[&y])[7], out[7], 1e-12);
    }

    #[test]
    fn hand_autocorr() {
        // 1,2,3,4: m = 2.5, deviations -1.5,-.5,.5,1.5. Lag 1: (-.5)(-1.5) + (.5)(-.5) +
        // (1.5)(.5) = 1.25 over sum of squares 5 = 0.25.
        let out = run("autocorr", &[int(4), int(1)], &[&[1.0, 2.0, 3.0, 4.0]]);
        assert_eq!(out[3], 0.25);
        assert!(run("autocorr", &[int(4), int(1)], &[&[2.0; 4]])[3].is_nan());
        assert!((lookup("autocorr").unwrap().make)(&[int(4), int(3)], &[0], &[1]).is_err());
    }

    #[test]
    fn hand_rolling_var_hist() {
        // Sorted window: -0.05, -0.02, -0.01, 0.01, 0.03 (p = 5).
        let x = [-0.05, 0.01, -0.02, 0.03, -0.01];
        // q = 0.25: position 0.25 * 4 = 1 -> -0.02, so VaR 0.02.
        near(run("rolling_var_hist", &[int(5), fl(0.25)], &[&x])[4], 0.02, 1e-15);
        // q = 0.3: position 1.2 -> -0.02 + 0.2 * 0.01 = -0.018.
        near(run("rolling_var_hist", &[int(5), fl(0.3)], &[&x])[4], 0.018, 1e-12);
        // q = 0 is the worst loss, q = 1 the best outcome negated.
        near(run("rolling_var_hist", &[int(5), fl(0.0)], &[&x])[4], 0.05, 1e-15);
        near(run("rolling_var_hist", &[int(5), fl(1.0)], &[&x])[4], -0.03, 1e-15);
    }

    #[test]
    fn hand_rolling_cvar_hist() {
        // Same window. q = 0.25: index 1, mean of -0.05 and -0.02 is -0.035, so 0.035.
        // q = 0.3: position 1.2, index 1 again.
        let x = [-0.05, 0.01, -0.02, 0.03, -0.01];
        near(run("rolling_cvar_hist", &[int(5), fl(0.25)], &[&x])[4], 0.035, 1e-15);
        near(run("rolling_cvar_hist", &[int(5), fl(0.3)], &[&x])[4], 0.035, 1e-12);
        // Never smaller than VaR at the same q.
        let v = run("rolling_var_hist", &[int(5), fl(0.3)], &[&x])[4];
        assert!(run("rolling_cvar_hist", &[int(5), fl(0.3)], &[&x])[4] >= v);
    }

    #[test]
    fn hand_max_drawdown() {
        // 100, 120, 90, 110, 80: peak 120 against 80 is the deepest, (120 - 80) / 120 = 1/3.
        let x = [100.0, 120.0, 90.0, 110.0, 80.0];
        near(run("max_drawdown", &[int(5)], &[&x])[4], 1.0 / 3.0, 1e-15);
        // A rising window has none; once the peak leaves the window the fall is forgotten.
        assert_eq!(run("max_drawdown", &[int(3)], &[&[1.0, 2.0, 3.0]])[2], 0.0);
        let out = run("max_drawdown", &[int(3)], &[&[100.0, 50.0, 60.0, 70.0]]);
        near(out[2], 0.5, 1e-15);
        assert_eq!(out[3], 0.0);
        assert!(run("max_drawdown", &[int(2)], &[&[1.0, 0.0, 2.0]])[1].is_nan());
    }

    #[test]
    fn hand_downside_dev() {
        // Below zero: -0.01 and -0.03; squares 1e-4 + 9e-4 = 1e-3 over p = 4.
        let x = [0.02, -0.01, -0.03, 0.01];
        near(run("downside_dev", &[int(4), fl(0.0)], &[&x])[3], (1e-3f64 / 4.0).sqrt(), 1e-15);
        // mar = 0.01: x - mar = .01, -.02, -.04, 0 -> squares 0, 4e-4, 16e-4, 0.
        near(
            run("downside_dev", &[int(4), fl(0.01)], &[&x])[3],
            ((4e-4 + 16e-4) / 4.0f64).sqrt(),
            1e-15,
        );
    }

    #[test]
    fn hand_rolling_sharpe() {
        // 0.01, 0.03, 0.02: mean 0.02, deviations -.01, 0, .01, variance 2e-4 / 2 = 1e-4,
        // sd 0.01: ratio 2.
        let out = run("rolling_sharpe", &[int(3)], &[&[0.01, 0.03, 0.02]]);
        near(out[2], 2.0, 1e-12);
        assert!(run("rolling_sharpe", &[int(3)], &[&[0.01; 3]])[2].is_nan());
    }

    #[test]
    fn hand_true_range_pct() {
        // Bar 1 (previous close 100): H 105, L 99: max(6, 5, 1) / 100 = 0.06.
        // Bar 2 (previous close 102): H 110, L 106: max(4, 8, 4) / 102.
        let h = [100.0, 105.0, 110.0];
        let l = [100.0, 99.0, 106.0];
        let c = [100.0, 102.0, 108.0];
        let out = run("true_range_pct", &[], &[&h, &l, &c]);
        assert!(out[0].is_nan());
        near(out[1], 0.06, 1e-15);
        near(out[2], 8.0 / 102.0, 1e-15);
    }

    #[test]
    fn hand_gap() {
        // Open 105 over previous close 100: 0.05; open 99 over close 110: -0.1.
        let o = [100.0, 105.0, 99.0];
        let c = [100.0, 110.0, 101.0];
        let out = run("gap", &[], &[&o, &c]);
        assert!(out[0].is_nan());
        near(out[1], 0.05, 1e-15);
        near(out[2], -0.1, 1e-15);
    }

    #[test]
    fn hand_body_pct() {
        // (110 - 105) / 105 and (101 - 99) / 99.
        let o = [100.0, 105.0, 99.0];
        let c = [100.0, 110.0, 101.0];
        let out = run("body_pct", &[], &[&o, &c]);
        assert_eq!(out[0], 0.0);
        near(out[1], 5.0 / 105.0, 1e-15);
        near(out[2], 2.0 / 99.0, 1e-15);
    }

    #[test]
    fn hand_range_pos() {
        // p = 3 over bars 0..=2: highest high 12, lowest low 7, close 10: (10 - 7) / 5 = 0.6.
        let h = [10.0, 12.0, 11.0, 11.0];
        let l = [8.0, 9.0, 7.0, 9.0];
        let c = [9.0, 10.0, 10.0, 9.5];
        let out = run("range_pos", &[int(3)], &[&h, &l, &c]);
        assert!(out[1].is_nan());
        near(out[2], 0.6, 1e-15);
        // Bar 3's window is bars 1..=3: highs max 12, lows min 7: (9.5 - 7) / 5 = 0.5.
        near(out[3], 0.5, 1e-15);
        // A flat range is NaN.
        let f = [5.0; 3];
        assert!(run("range_pos", &[int(3)], &[&f, &f, &f])[2].is_nan());
    }

    fn cases() -> Vec<(&'static str, Vec<ParamVal>)> {
        vec![
            ("vol_close", vec![int(5)]),
            ("vol_parkinson", vec![int(5)]),
            ("vol_garman_klass", vec![int(5)]),
            ("vol_rogers_satchell", vec![int(5)]),
            ("vol_yang_zhang", vec![int(5)]),
            ("rolling_beta", vec![int(6)]),
            ("rolling_alpha", vec![int(6)]),
            ("rolling_r2", vec![int(6)]),
            ("rolling_residual", vec![int(6)]),
            ("rolling_corr_spearman", vec![int(6)]),
            ("half_life", vec![int(10)]),
            ("fracdiff", vec![fl(0.4), fl(0.01)]),
            ("hurst_rs", vec![int(16)]),
            ("autocorr", vec![int(10), int(2)]),
            ("rolling_var_hist", vec![int(10), fl(0.1)]),
            ("rolling_cvar_hist", vec![int(10), fl(0.1)]),
            ("max_drawdown", vec![int(7)]),
            ("downside_dev", vec![int(7), fl(100.0)]),
            ("rolling_sharpe", vec![int(7)]),
            ("true_range_pct", vec![]),
            ("gap", vec![]),
            ("body_pct", vec![]),
            ("range_pos", vec![int(6)]),
        ]
    }

    #[test]
    fn every_function_has_a_case_and_a_hand_test_name() {
        let names: Vec<&str> = cases().iter().map(|c| c.0).collect();
        for d in DEFS {
            assert!(names.contains(&d.name), "{} missing from cases()", d.name);
            match d.reference {
                Reference::Hand(t) => assert_eq!(t, format!("hand_{}", d.name)),
                Reference::Fixture => panic!("{} has no hand test", d.name),
            }
        }
        assert_eq!(DEFS.len(), names.len());
    }

    #[test]
    fn batch_and_streaming_are_bit_identical_with_nans() {
        let mut rng = Rng(7);
        for (name, params) in cases() {
            let arity = lookup(name).unwrap().inputs.len();
            let cols: Vec<Vec<f64>> = (0..arity)
                .map(|_| {
                    (0..220)
                        .map(|_| match rng.next() % 20 {
                            0 => f64::NAN,
                            1 => f64::INFINITY,
                            _ => 50.0 + 100.0 * rng.unif(),
                        })
                        .collect()
                })
                .collect();
            let views: Vec<&[f64]> = cols.iter().map(|c| &c[..]).collect();
            let b: Vec<u64> = run(name, &params, &views).into_iter().map(canon_bits).collect();
            let s: Vec<u64> = stream(name, &params, &views).into_iter().map(canon_bits).collect();
            assert_eq!(b, s, "{name}");
        }
    }

    #[test]
    fn warmup_formula_matches_first_non_nan() {
        let mut rng = Rng(9);
        for (name, mut params) in cases() {
            if name == "fracdiff" {
                params = vec![fl(0.4), fl(0.05)];
            }
            let def = lookup(name).unwrap();
            let cols: Vec<Vec<f64>> = (0..def.inputs.len())
                .map(|_| (0..120).map(|_| 50.0 + 100.0 * rng.unif()).collect())
                .collect();
            let views: Vec<&[f64]> = cols.iter().map(|c| &c[..]).collect();
            let out = run(name, &params, &views);
            let first = out.iter().position(|v| !v.is_nan());
            // Random data can make a result undefined (half_life with phi outside (0, 1), OHLC
            // that is not a valid candle); there the formula only bounds the first value.
            let first = first.expect(name);
            let undefined_on_noise = [
                "half_life",
                "hurst_rs",
                "range_pos",
                "vol_garman_klass",
                "vol_rogers_satchell",
                "vol_yang_zhang",
            ];
            if undefined_on_noise.contains(&name) {
                assert!(first >= (def.warmup)(&params), "{name}");
            } else {
                assert_eq!(first, (def.warmup)(&params), "{name}");
            }
        }
    }
}
