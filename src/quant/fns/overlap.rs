//! Overlap studies. `sma`, `ema`, `wma`, `wilder` and `bbands` replicate the streaming cores in
//! `indicators::streaming` operation for operation, so digests and live agree. The rest follow
//! TA-Lib's conventions on clean data (each cascade stage is seeded with the SMA of the previous
//! stage's first `period` outputs; SAR's first direction comes from the minus directional move)
//! and the expression NaN policy on top.
//!
//! [`MaCore`] and [`MaKind`] are the one-step moving averages that other kernels embed
//! (TA-Lib's `matype`).

use crate::quant::fns::hilbert::MamaSt;
use crate::quant::kernel::{Adapter, Hist, Kernel, Node};
use crate::quant::registry::{
    int_param, regs, Domain, Family, FnDef, InputDef, Invariance, Legacy, Memory, NanPolicy,
    NanRule, OutKind, ParamDef, ParamKind, ParamVal, Reference, Relation,
};
use crate::quant::window::{RollExt, ShiftedWin};

const MAX_PERIOD: i64 = 1 << 24;

/// Same arithmetic as `StreamingSma`: add the new value, then subtract the one leaving; a
/// non-finite value is counted, not summed.
pub struct Sma {
    p: usize,
    sum: f64,
    bad: usize,
    n: u64,
}

impl Sma {
    pub fn new(p: usize) -> Self {
        Self { p: p.max(1), sum: 0.0, bad: 0, n: 0 }
    }
}

impl Kernel<1, 1> for Sma {
    fn heap_bytes(&self) -> usize {
        0
    }

    fn lookback(&self) -> usize {
        self.p
    }

    #[inline(always)]
    fn step<H: Hist<1>>(&mut self, x: [f64; 1], h: &H) -> [f64; 1] {
        let x = x[0];
        self.n += 1;
        // A non-finite value is counted and adds 0, so the sum's dependency chain holds two
        // additions and no select on the finiteness of an input (same bits as skipping it).
        let fx = x.is_finite();
        self.sum += if fx { x } else { 0.0 };
        self.bad += usize::from(!fx);
        if self.n > self.p as u64 {
            let old = h.ago(0, self.p);
            let fo = old.is_finite();
            self.sum -= if fo { old } else { 0.0 };
            self.bad -= usize::from(!fo);
        }
        if self.n < self.p as u64 || self.bad > 0 {
            [f64::NAN]
        } else {
            [self.sum / self.p as f64]
        }
    }
}

/// Same arithmetic as `StreamingEma`, SMA-seeded. Differs on non-finite input by the expression
/// NaN policy: a NaN neither seeds nor advances the state (the legacy core seeds NaN forever).
pub struct Ema {
    st: EmaSt,
}

impl Ema {
    pub fn new(p: usize) -> Self {
        Self { st: EmaSt::new(p) }
    }
}

impl Kernel<1, 1> for Ema {
    fn heap_bytes(&self) -> usize {
        0
    }

    fn lookback(&self) -> usize {
        0
    }

    #[inline(always)]
    fn step<H: Hist<1>>(&mut self, x: [f64; 1], _h: &H) -> [f64; 1] {
        [self.st.step(x[0])]
    }
}

/// The last `len` values pushed, in a ring allocated once.
struct Ring {
    buf: Box<[f64]>,
    head: usize,
    n: u64,
}

impl Ring {
    /// Heap owned, at capacity.
    fn heap_bytes(&self) -> usize {
        std::mem::size_of_val(&*self.buf)
    }
}

impl Ring {
    fn new(len: usize) -> Self {
        assert!(len >= 1, "a window of at least 1");
        Self { buf: vec![f64::NAN; len].into_boxed_slice(), head: 0, n: 0 }
    }

    /// Push `x`; returns the value that left (pushed `len` pushes ago). It is meaningful only
    /// when `leaving()` is true afterwards; before that it is the NaN the ring started with.
    fn push(&mut self, x: f64) -> f64 {
        let old = self.buf[self.head];
        self.buf[self.head] = x;
        self.head = (self.head + 1) % self.buf.len();
        self.n += 1;
        old
    }

    /// True when the push just made left a value out of the ring.
    fn leaving(&self) -> bool {
        self.n > self.buf.len() as u64
    }

    fn full(&self) -> bool {
        self.n >= self.buf.len() as u64
    }

    /// The oldest value held (valid once full).
    fn oldest(&self) -> f64 {
        self.buf[self.head]
    }

    /// The `j`-th value held, oldest first (valid once full).
    #[inline(always)]
    fn at(&self, j: usize) -> f64 {
        self.buf[(self.head + j) % self.buf.len()]
    }
}

/// `Sma`'s arithmetic over an owned ring: add the new value, then subtract the one leaving; a
/// non-finite value is counted, not summed.
struct RingSma {
    p: usize,
    ring: Ring,
    sum: f64,
    bad: usize,
}

impl RingSma {
    /// Heap owned, at capacity.
    fn heap_bytes(&self) -> usize {
        self.ring.heap_bytes()
    }
}

impl RingSma {
    fn new(p: usize) -> Self {
        Self { p, ring: Ring::new(p), sum: 0.0, bad: 0 }
    }

    fn push(&mut self, x: f64) -> f64 {
        // A non-finite value is counted and adds 0 (no select on the sum's dependency chain).
        let fx = x.is_finite();
        self.sum += if fx { x } else { 0.0 };
        self.bad += usize::from(!fx);
        let old = self.ring.push(x);
        if self.ring.leaving() {
            let fo = old.is_finite();
            self.sum -= if fo { old } else { 0.0 };
            self.bad -= usize::from(!fo);
        }
        if !self.ring.full() || self.bad > 0 {
            f64::NAN
        } else {
            self.sum / self.p as f64
        }
    }
}

/// `Ema`'s arithmetic: SMA-seeded; a non-finite value neither seeds nor advances the state.
struct EmaSt {
    p: usize,
    alpha: f64,
    seeded: Option<f64>,
    seed_sum: f64,
    seen: usize,
}

impl EmaSt {
    fn new(p: usize) -> Self {
        let p = p.max(1);
        Self::with_alpha(p, 2.0 / (p as f64 + 1.0))
    }

    fn with_alpha(p: usize, alpha: f64) -> Self {
        Self { p: p.max(1), alpha, seeded: None, seed_sum: 0.0, seen: 0 }
    }

    fn step(&mut self, x: f64) -> f64 {
        if !x.is_finite() {
            return f64::NAN;
        }
        match self.seeded {
            Some(prev) => {
                let v = self.alpha * x + (1.0 - self.alpha) * prev;
                self.seeded = Some(v);
                v
            }
            None => {
                self.seed_sum += x;
                self.seen += 1;
                if self.seen == self.p {
                    let v = self.seed_sum / self.p as f64;
                    self.seeded = Some(v);
                    v
                } else {
                    f64::NAN
                }
            }
        }
    }
}

const KAMA_MAX: f64 = 2.0 / 31.0;
const KAMA_DIFF: f64 = 2.0 / 3.0 - KAMA_MAX;
/// A window whose path length is below this counts as flat (efficiency ratio 1). The rolling sum
/// of a flat window after a volatile one is a rounding residue, not 0; TA-Lib's band is 1e-14,
/// measured: it agrees there with this and not with an exact-zero test or a 1e-8 band.
const KAMA_FLAT: f64 = 1e-14;

/// Kaufman's adaptive average: efficiency ratio over `p` bars, smoothing constant between a
/// 2-bar and a 30-bar EMA, seeded with the bar before the first output. A non-finite value is
/// not a bar: it neither enters the windows nor moves the state.
struct KamaSt {
    p: usize,
    xs: Ring,
    ds: Ring,
    c: u64,
    last: f64,
    sum: f64,
    prev: f64,
}

impl KamaSt {
    /// Heap owned, at capacity.
    fn heap_bytes(&self) -> usize {
        self.xs.heap_bytes() + self.ds.heap_bytes()
    }
}

impl KamaSt {
    fn new(p: usize) -> Self {
        Self {
            p,
            xs: Ring::new(p + 1),
            ds: Ring::new(p),
            c: 0,
            last: f64::NAN,
            sum: 0.0,
            prev: f64::NAN,
        }
    }

    fn step(&mut self, x: f64) -> f64 {
        if !x.is_finite() {
            return f64::NAN;
        }
        let c = self.c;
        self.c += 1;
        let before = self.last;
        if c >= 1 {
            let d = (x - before).abs();
            let gone = self.ds.push(d);
            if c > self.p as u64 {
                self.sum -= gone;
            }
            self.sum += d;
        }
        self.last = x;
        self.xs.push(x);
        if c < self.p as u64 {
            return f64::NAN;
        }
        if c == self.p as u64 {
            self.prev = before;
        }
        let roc = x - self.xs.oldest();
        let er = if self.sum <= roc || self.sum < KAMA_FLAT { 1.0 } else { (roc / self.sum).abs() };
        let s = er * KAMA_DIFF + KAMA_MAX;
        self.prev = (x - self.prev) * (s * s) + self.prev;
        self.prev
    }
}

struct T3St {
    e: [EmaSt; 6],
    c: [f64; 4],
}

impl T3St {
    fn new(p: usize, a: f64) -> Self {
        let (a2, a3) = (a * a, a * a * a);
        Self {
            e: std::array::from_fn(|_| EmaSt::new(p)),
            c: [
                -a3,
                3.0 * a2 + 3.0 * a3,
                -6.0 * a2 - 3.0 * a - 3.0 * a3,
                1.0 + 3.0 * a + a3 + 3.0 * a2,
            ],
        }
    }

    fn step(&mut self, x: f64) -> f64 {
        let mut v = [f64::NAN; 6];
        let mut x = x;
        for (st, out) in self.e.iter_mut().zip(v.iter_mut()) {
            x = st.step(x);
            if x.is_nan() {
                return f64::NAN;
            }
            *out = x;
        }
        self.c[0] * v[5] + self.c[1] * v[4] + self.c[2] * v[3] + self.c[3] * v[2]
    }
}

/// TA-Lib's `matype`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MaKind {
    Sma,
    Ema,
    Wma,
    Dema,
    Tema,
    Trima,
    Kama,
    Mama,
    T3,
}

impl MaKind {
    /// `matype` as TA-Lib numbers it: 0 SMA, 1 EMA, 2 WMA, 3 DEMA, 4 TEMA, 5 TRIMA, 6 KAMA,
    /// 7 MAMA, 8 T3.
    pub(crate) fn from_code(code: i64) -> Result<Self, String> {
        Ok(match code {
            0 => Self::Sma,
            1 => Self::Ema,
            2 => Self::Wma,
            3 => Self::Dema,
            4 => Self::Tema,
            5 => Self::Trima,
            6 => Self::Kama,
            7 => Self::Mama,
            8 => Self::T3,
            _ => return Err(format!("matype must be 0 to 8, got {code}")),
        })
    }

    /// Leading NaN outputs of a clean series at `period`.
    pub(crate) fn lookback(self, period: usize) -> usize {
        if period <= 1 {
            return 0;
        }
        let p1 = period - 1;
        match self {
            Self::Sma | Self::Ema | Self::Wma | Self::Trima => p1,
            Self::Dema => 2 * p1,
            Self::Tema => 3 * p1,
            Self::T3 => 6 * p1,
            Self::Kama => period,
            // TA-Lib's MAMA lookback; it ignores the period.
            Self::Mama => 32,
        }
    }

    /// Bars until the influence of the seed falls below `eps` (0 for the finite-memory kinds).
    pub(crate) fn settle(self, period: usize, eps: f64) -> usize {
        let alpha = 2.0 / (period as f64 + 1.0);
        match self {
            Self::Sma | Self::Wma | Self::Trima => 0,
            Self::Ema => decay_bars(1.0 - alpha, 1, eps),
            Self::Dema => decay_bars(1.0 - alpha, 2, eps),
            Self::Tema => decay_bars(1.0 - alpha, 3, eps),
            Self::T3 => decay_bars(1.0 - alpha, 6, eps),
            // The smoothing constant is smallest when the efficiency ratio is 0.
            Self::Kama => decay_bars(1.0 - KAMA_MAX * KAMA_MAX, 1, eps),
            // FAMA at the slow limit 0.05 smooths by half of it; the Hilbert state is slower
            // still, so this is a guide (see `mama`).
            Self::Mama => 32 + decay_bars(1.0 - 0.025, 1, eps),
        }
    }
}

/// Bars `k` until `(k + 1)^(stages - 1) * rho^k <= eps`, the bound on the seed's influence after
/// `stages` cascaded EMAs of decay `rho`.
fn decay_bars(rho: f64, stages: u32, eps: f64) -> usize {
    let eps = eps.max(1e-300);
    if rho <= 0.0 {
        return 0;
    }
    let (lr, le) = (rho.ln(), eps.ln());
    let mut k = 0usize;
    while k < (1 << 26) && f64::from(stages - 1) * ((k + 1) as f64).ln() + k as f64 * lr > le {
        k += 1;
    }
    k
}

enum Inner {
    /// Period 1: the value itself (TA-Lib's MA copies the input).
    Identity,
    Sma(RingSma),
    Ema(EmaSt),
    Wma(Ring, ShiftedWin),
    Dema([EmaSt; 2]),
    Tema([EmaSt; 3]),
    Trima(RingSma, RingSma),
    Kama(KamaSt),
    T3(Box<T3St>),
    /// TA-Lib's MA with matype 7: fast and slow limits 0.5 and 0.05 whatever the period.
    Mama(Box<MamaSt>),
}

impl Inner {
    /// Heap owned, at capacity.
    fn heap_bytes(&self) -> usize {
        match self {
            Inner::Identity | Inner::Ema(_) | Inner::Dema(_) | Inner::Tema(_) => 0,
            Inner::Sma(r) => r.heap_bytes(),
            Inner::Wma(r, _) => r.heap_bytes(),
            Inner::Trima(a, b) => a.heap_bytes() + b.heap_bytes(),
            Inner::Kama(k) => k.heap_bytes(),
            Inner::T3(_) => std::mem::size_of::<T3St>(),
            Inner::Mama(_) => std::mem::size_of::<MamaSt>(),
        }
    }
}

/// A moving average of any TA-Lib `matype` as a one-step state machine over a single series, for
/// use inside another kernel (BBANDS, MA, MAVP here; STOCH, MACDEXT, APO and PPO in the momentum
/// family). It owns its history, so it needs no `Hist`. Windowed kinds (SMA, WMA, TRIMA) output
/// NaN while a non-finite value is in the window and then recover; recursive kinds (EMA, DEMA,
/// TEMA, KAMA, T3) output NaN on a non-finite value without advancing.
pub(crate) struct MaCore {
    inner: Inner,
    warmed: bool,
}

impl MaCore {
    /// Heap owned, at capacity.
    pub(crate) fn heap_bytes(&self) -> usize {
        self.inner.heap_bytes()
    }
}

impl MaCore {
    /// T3 uses TA-Lib's `vfactor` of 0.7 here.
    pub(crate) fn new(kind: MaKind, period: usize) -> Result<Self, String> {
        Self::with_vfactor(kind, period, 0.7)
    }

    pub(crate) fn with_vfactor(kind: MaKind, period: usize, vfactor: f64) -> Result<Self, String> {
        if period == 0 {
            return Err("moving average period must be at least 1".into());
        }
        let inner = match kind {
            _ if period == 1 => Inner::Identity,
            MaKind::Mama => Inner::Mama(Box::new(MamaSt::new(0.5, 0.05))),
            MaKind::Sma => Inner::Sma(RingSma::new(period)),
            MaKind::Ema => Inner::Ema(EmaSt::new(period)),
            MaKind::Wma => Inner::Wma(Ring::new(period), ShiftedWin::new(period)),
            MaKind::Dema => Inner::Dema([EmaSt::new(period), EmaSt::new(period)]),
            MaKind::Tema => {
                Inner::Tema([EmaSt::new(period), EmaSt::new(period), EmaSt::new(period)])
            }
            MaKind::Trima => {
                // Odd n: two windows of (n + 1) / 2. Even n: n / 2 then n / 2 + 1.
                let (p1, p2) = if period % 2 == 1 {
                    (period.div_ceil(2), period.div_ceil(2))
                } else {
                    (period / 2, period / 2 + 1)
                };
                Inner::Trima(RingSma::new(p1), RingSma::new(p2))
            }
            MaKind::Kama => Inner::Kama(KamaSt::new(period)),
            MaKind::T3 => Inner::T3(Box::new(T3St::new(period, vfactor))),
        };
        Ok(Self { inner, warmed: false })
    }

    pub(crate) fn step(&mut self, x: f64) -> f64 {
        let v = match &mut self.inner {
            Inner::Identity => {
                if x.is_finite() {
                    x
                } else {
                    f64::NAN
                }
            }
            Inner::Sma(s) => s.push(x),
            Inner::Ema(e) => e.step(x),
            Inner::Wma(r, w) => {
                let old = r.push(x);
                if w.push(x, old, |j| r.at(j)) {
                    w.weighted_mean()
                } else {
                    f64::NAN
                }
            }
            Inner::Dema(e) => {
                let v1 = e[0].step(x);
                if v1.is_nan() {
                    return f64::NAN;
                }
                let v2 = e[1].step(v1);
                2.0 * v1 - v2
            }
            Inner::Tema(e) => {
                let v1 = e[0].step(x);
                if v1.is_nan() {
                    return f64::NAN;
                }
                let v2 = e[1].step(v1);
                if v2.is_nan() {
                    return f64::NAN;
                }
                let v3 = e[2].step(v2);
                v3 + (3.0 * v1 - 3.0 * v2)
            }
            Inner::Trima(a, b) => {
                let v = a.push(x);
                b.push(v)
            }
            Inner::Kama(k) => k.step(x),
            Inner::T3(t) => t.step(x),
            Inner::Mama(m) => m.step(x),
        };
        self.warmed |= v.is_finite();
        v
    }

    /// True once the average has produced a value, and stays true through later NaN gaps.
    pub(crate) fn warmed(&self) -> bool {
        self.warmed
    }
}

/// One moving average as a kernel; the node behind `wma`, `dema`, `tema`, `trima`, `kama`, `t3`
/// and `ma`.
pub struct MaFn {
    core: MaCore,
}

impl Kernel<1, 1> for MaFn {
    fn heap_bytes(&self) -> usize {
        self.core.heap_bytes()
    }

    fn lookback(&self) -> usize {
        0
    }

    #[inline(always)]
    fn step<H: Hist<1>>(&mut self, x: [f64; 1], _h: &H) -> [f64; 1] {
        [self.core.step(x[0])]
    }
}

/// Wilder smoothing: EMA with alpha = 1/period, SMA-seeded (`StreamingEma::wilder`).
pub struct Wilder {
    st: EmaSt,
}

impl Kernel<1, 1> for Wilder {
    fn heap_bytes(&self) -> usize {
        0
    }

    fn lookback(&self) -> usize {
        0
    }

    #[inline(always)]
    fn step<H: Hist<1>>(&mut self, x: [f64; 1], _h: &H) -> [f64; 1] {
        [self.st.step(x[0])]
    }
}

/// Bollinger bands. The deviation is the population deviation of the window around its simple
/// mean (as `StreamingBollinger`), whatever the middle band's `matype`.
pub struct BBands {
    window: RingSma,
    var: ShiftedWin,
    ma: Option<MaCore>,
    up: f64,
    dn: f64,
}

impl Kernel<1, 3> for BBands {
    fn heap_bytes(&self) -> usize {
        self.window.heap_bytes() + self.ma.as_ref().map_or(0, MaCore::heap_bytes)
    }

    fn lookback(&self) -> usize {
        0
    }

    #[inline(always)]
    fn step<H: Hist<1>>(&mut self, x: [f64; 1], _h: &H) -> [f64; 3] {
        let x = x[0];
        let old = if self.window.ring.full() { self.window.ring.oldest() } else { f64::NAN };
        let mean = self.window.push(x);
        let clean = self.var.push(x, old, |j| self.window.ring.at(j));
        let mid = match &mut self.ma {
            None => mean,
            Some(m) => m.step(x),
        };
        if mid.is_nan() || mean.is_nan() || !clean {
            return [f64::NAN, mid, f64::NAN];
        }
        let sd = (self.var.m2() / self.window.p as f64).sqrt();
        [mid + self.up * sd, mid, mid - self.dn * sd]
    }
}

/// Moving average with a per-bar period, seeded as TA-Lib's MAVP seeds it: the average of
/// period `q` starts at bar `S - lookback(q)`, `S` being the longest period's lookback, so every
/// period produces its first value at bar `S` and a recursive kind (EMA, DEMA, TEMA, KAMA, T3)
/// carries no state from before it. For SMA, WMA and TRIMA this is the same as running from bar 0.
pub struct Mavp {
    min: usize,
    max: usize,
    cores: Vec<MaCore>,
    /// The bar at which each core takes its first value.
    starts: Vec<usize>,
    n: usize,
}

impl Kernel<2, 1> for Mavp {
    fn heap_bytes(&self) -> usize {
        self.cores.capacity() * std::mem::size_of::<MaCore>()
            + self.cores.iter().map(MaCore::heap_bytes).sum::<usize>()
            + self.starts.capacity() * std::mem::size_of::<usize>()
    }

    fn lookback(&self) -> usize {
        0
    }

    #[inline(always)]
    fn step<H: Hist<2>>(&mut self, x: [f64; 2], _h: &H) -> [f64; 1] {
        let [v, per] = x;
        let pick = if per.is_finite() {
            Some((per as i64).clamp(self.min as i64, self.max as i64) as usize - self.min)
        } else {
            None
        };
        let mut out = f64::NAN;
        for (i, c) in self.cores.iter_mut().enumerate() {
            if self.n < self.starts[i] {
                continue;
            }
            let r = c.step(v);
            if pick == Some(i) {
                out = r;
            }
        }
        self.n += 1;
        // TA-Lib's first output is the warm-up of the longest period.
        if self.cores.last().is_some_and(MaCore::warmed) {
            [out]
        } else {
            [f64::NAN]
        }
    }
}

struct RollMid {
    hi: RollExt,
    lo: RollExt,
}

impl RollMid {
    /// Heap owned, at capacity.
    fn heap_bytes(&self) -> usize {
        self.hi.heap_bytes() + self.lo.heap_bytes()
    }
}

impl RollMid {
    fn new(p: usize) -> Self {
        Self { hi: RollExt::new(p, true), lo: RollExt::new(p, false) }
    }

    fn step(&mut self, hi: f64, lo: f64) -> f64 {
        (self.hi.push(hi) + self.lo.push(lo)) / 2.0
    }
}

/// `midpoint`: (highest + lowest) / 2 of one series.
pub struct Midpoint(RollMid);

impl Kernel<1, 1> for Midpoint {
    fn heap_bytes(&self) -> usize {
        self.0.heap_bytes()
    }

    fn lookback(&self) -> usize {
        0
    }

    #[inline(always)]
    fn step<H: Hist<1>>(&mut self, x: [f64; 1], _h: &H) -> [f64; 1] {
        [self.0.step(x[0], x[0])]
    }
}

/// `midprice`: (highest high + lowest low) / 2.
pub struct Midprice(RollMid);

impl Kernel<2, 1> for Midprice {
    fn heap_bytes(&self) -> usize {
        self.0.heap_bytes()
    }

    fn lookback(&self) -> usize {
        0
    }

    #[inline(always)]
    fn step<H: Hist<2>>(&mut self, x: [f64; 2], _h: &H) -> [f64; 1] {
        [self.0.step(x[0], x[1])]
    }
}

/// Parabolic SAR, extended form; `sar` is the case start 0, offset 0, equal long and short
/// factors, with short values reported positive. The first bar only records its range. The
/// second sets the direction from the minus directional move of the first two bars (short when
/// positive) and starts from the first bar's opposite extreme, then runs the update with the
/// second bar standing as its own previous bar. A bar with a non-finite high or low is skipped.
pub struct Sar {
    start: f64,
    offset: f64,
    af_init: [f64; 2],
    af_step: [f64; 2],
    af_max: [f64; 2],
    signed: bool,
    n: u8,
    long: bool,
    ep: f64,
    sar: f64,
    af: [f64; 2],
    prev_hi: f64,
    prev_lo: f64,
}

impl Sar {
    /// `init`, `step` and `max` are `[long, short]`. An init or step above its max is cut to it.
    fn new(
        start: f64,
        offset: f64,
        init: [f64; 2],
        step: [f64; 2],
        max: [f64; 2],
        signed: bool,
    ) -> Self {
        let cut = |v: [f64; 2]| [v[0].min(max[0]), v[1].min(max[1])];
        let (af_init, af_step) = (cut(init), cut(step));
        Self {
            start,
            offset,
            af_init,
            af_step,
            af_max: max,
            signed,
            n: 0,
            long: true,
            ep: f64::NAN,
            sar: f64::NAN,
            af: af_init,
            prev_hi: f64::NAN,
            prev_lo: f64::NAN,
        }
    }

    fn step_bar(&mut self, hi: f64, lo: f64) -> f64 {
        match self.n {
            0 => {
                self.n = 1;
                self.prev_hi = hi;
                self.prev_lo = lo;
                return f64::NAN;
            }
            1 => {
                self.n = 2;
                if self.start == 0.0 {
                    let (dp, dm) = (hi - self.prev_hi, self.prev_lo - lo);
                    let minus = if dm > 0.0 && dp < dm { dm } else { 0.0 };
                    self.long = minus <= 0.0;
                    if self.long {
                        (self.ep, self.sar) = (hi, self.prev_lo);
                    } else {
                        (self.ep, self.sar) = (lo, self.prev_hi);
                    }
                } else if self.start > 0.0 {
                    (self.long, self.ep, self.sar) = (true, hi, self.start);
                } else {
                    (self.long, self.ep, self.sar) = (false, lo, -self.start);
                }
                self.prev_hi = hi;
                self.prev_lo = lo;
            }
            _ => {}
        }
        let (ph, pl) = (self.prev_hi, self.prev_lo);
        let out;
        if self.long {
            if lo <= self.sar {
                self.long = false;
                self.sar = self.ep.max(ph).max(hi);
                if self.offset != 0.0 {
                    self.sar += self.sar * self.offset;
                }
                out = if self.signed { -self.sar } else { self.sar };
                self.af[1] = self.af_init[1];
                self.ep = lo;
                self.sar += self.af[1] * (self.ep - self.sar);
                self.sar = self.sar.max(ph).max(hi);
            } else {
                out = self.sar;
                if hi > self.ep {
                    self.ep = hi;
                    self.af[0] = (self.af[0] + self.af_step[0]).min(self.af_max[0]);
                }
                self.sar += self.af[0] * (self.ep - self.sar);
                self.sar = self.sar.min(pl).min(lo);
            }
        } else if hi >= self.sar {
            self.long = true;
            self.sar = self.ep.min(pl).min(lo);
            if self.offset != 0.0 {
                self.sar -= self.sar * self.offset;
            }
            out = self.sar;
            self.af[0] = self.af_init[0];
            self.ep = hi;
            self.sar += self.af[0] * (self.ep - self.sar);
            self.sar = self.sar.min(pl).min(lo);
        } else {
            out = if self.signed { -self.sar } else { self.sar };
            if lo < self.ep {
                self.ep = lo;
                self.af[1] = (self.af[1] + self.af_step[1]).min(self.af_max[1]);
            }
            self.sar += self.af[1] * (self.ep - self.sar);
            self.sar = self.sar.max(ph).max(hi);
        }
        self.prev_hi = hi;
        self.prev_lo = lo;
        out
    }
}

impl Kernel<2, 1> for Sar {
    fn heap_bytes(&self) -> usize {
        0
    }

    fn lookback(&self) -> usize {
        0
    }

    #[inline(always)]
    fn step<H: Hist<2>>(&mut self, x: [f64; 2], _h: &H) -> [f64; 1] {
        if !x[0].is_finite() || !x[1].is_finite() {
            return [f64::NAN];
        }
        [self.step_bar(x[0], x[1])]
    }
}

fn float_param(p: &[ParamVal], i: usize) -> Result<f64, String> {
    match p.get(i) {
        Some(ParamVal::Float(v)) => Ok(*v),
        Some(ParamVal::Int(v)) => Ok(*v as f64),
        other => Err(format!("parameter {i} must be a number, got {other:?}")),
    }
}

fn kind_param(p: &[ParamVal], i: usize) -> Result<MaKind, String> {
    MaKind::from_code(int_param(p, i)? as i64)
}

const INPUT_X: &[InputDef] = &[InputDef { name: "x", default_field: Some("close") }];
const PERIOD: &[ParamDef] =
    &[ParamDef { name: "period", kind: ParamKind::Int { min: 1, max: MAX_PERIOD }, default: None }];

fn warmup_period(p: &[ParamVal]) -> usize {
    int_param(p, 0).map_or(0, |v| v.saturating_sub(1))
}

fn make_sma(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = Sma::new(int_param(p, 0)?);
    Ok(Box::new(Adapter::new(k, regs::<1>(ins, "sma inputs")?, regs::<1>(outs, "sma outputs")?)))
}

fn make_ema(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = Ema::new(int_param(p, 0)?);
    Ok(Box::new(Adapter::new(k, regs::<1>(ins, "ema inputs")?, regs::<1>(outs, "ema outputs")?)))
}

fn ema_settle(p: &[ParamVal], eps: f64) -> usize {
    // Seed influence decays as (1 - alpha)^k.
    let period = int_param(p, 0).unwrap_or(1) as f64;
    let alpha = 2.0 / (period + 1.0);
    (eps.ln() / (1.0 - alpha).ln()).ceil().max(0.0) as usize
}

const INPUT_HL: &[InputDef] = &[
    InputDef { name: "high", default_field: Some("high") },
    InputDef { name: "low", default_field: Some("low") },
];
const INPUT_X_PERIODS: &[InputDef] = &[
    InputDef { name: "x", default_field: Some("close") },
    InputDef { name: "periods", default_field: None },
];
const PERIOD_MIN2: &[ParamDef] =
    &[ParamDef { name: "period", kind: ParamKind::Int { min: 2, max: MAX_PERIOD }, default: None }];
const MATYPE: ParamDef = ParamDef {
    name: "matype",
    kind: ParamKind::Int { min: 0, max: 8 },
    default: Some(ParamVal::Int(0)),
};
const PERIOD_MATYPE: &[ParamDef] = &[PERIOD[0], MATYPE];
const T3_PARAMS: &[ParamDef] = &[
    PERIOD[0],
    ParamDef {
        name: "vfactor",
        kind: ParamKind::Float { min: 0.0, max: 1.0 },
        default: Some(ParamVal::Float(0.7)),
    },
];
const BBANDS_PARAMS: &[ParamDef] = &[
    PERIOD[0],
    ParamDef {
        name: "nbdevup",
        kind: ParamKind::Float { min: -1e6, max: 1e6 },
        default: Some(ParamVal::Float(2.0)),
    },
    ParamDef {
        name: "nbdevdn",
        kind: ParamKind::Float { min: -1e6, max: 1e6 },
        default: Some(ParamVal::Float(2.0)),
    },
    MATYPE,
];
/// Largest period of `mavp`: every period in the range keeps a running average.
const MAVP_MAX: i64 = 1000;
const MAVP_PARAMS: &[ParamDef] = &[
    ParamDef {
        name: "minperiod",
        kind: ParamKind::Int { min: 1, max: MAVP_MAX },
        default: Some(ParamVal::Int(2)),
    },
    ParamDef {
        name: "maxperiod",
        kind: ParamKind::Int { min: 1, max: MAVP_MAX },
        default: Some(ParamVal::Int(30)),
    },
    MATYPE,
];
const fn accel(name: &'static str, default: f64) -> ParamDef {
    ParamDef {
        name,
        kind: ParamKind::Float { min: 0.0, max: 1e6 },
        default: Some(ParamVal::Float(default)),
    }
}
const SAR_PARAMS: &[ParamDef] = &[accel("acceleration", 0.02), accel("maximum", 0.2)];
const SAREXT_PARAMS: &[ParamDef] = &[
    ParamDef {
        name: "startvalue",
        kind: ParamKind::Float { min: -1e15, max: 1e15 },
        default: Some(ParamVal::Float(0.0)),
    },
    ParamDef {
        name: "offsetonreverse",
        kind: ParamKind::Float { min: -1e6, max: 1e6 },
        default: Some(ParamVal::Float(0.0)),
    },
    accel("accelerationinitlong", 0.02),
    accel("accelerationlong", 0.02),
    accel("accelerationmaxlong", 0.2),
    accel("accelerationinitshort", 0.02),
    accel("accelerationshort", 0.02),
    accel("accelerationmaxshort", 0.2),
];

fn dema_warmup(p: &[ParamVal]) -> usize {
    int_param(p, 0).map_or(0, |v| MaKind::Dema.lookback(v))
}
fn tema_warmup(p: &[ParamVal]) -> usize {
    int_param(p, 0).map_or(0, |v| MaKind::Tema.lookback(v))
}
fn trima_warmup(p: &[ParamVal]) -> usize {
    int_param(p, 0).map_or(0, |v| MaKind::Trima.lookback(v))
}
fn kama_warmup(p: &[ParamVal]) -> usize {
    int_param(p, 0).map_or(0, |v| MaKind::Kama.lookback(v))
}
fn t3_warmup(p: &[ParamVal]) -> usize {
    int_param(p, 0).map_or(0, |v| MaKind::T3.lookback(v))
}

fn ma_warmup(p: &[ParamVal]) -> usize {
    match (int_param(p, 0), kind_param(p, 1)) {
        (Ok(period), Ok(kind)) => kind.lookback(period),
        _ => 0,
    }
}

fn bbands_warmup(p: &[ParamVal]) -> usize {
    match (int_param(p, 0), kind_param(p, 3)) {
        (Ok(period), Ok(kind)) => kind.lookback(period).max(period.saturating_sub(1)),
        _ => 0,
    }
}

fn mavp_warmup(p: &[ParamVal]) -> usize {
    match (int_param(p, 1), kind_param(p, 2)) {
        (Ok(max), Ok(kind)) => kind.lookback(max),
        _ => 0,
    }
}

fn sar_warmup(_: &[ParamVal]) -> usize {
    1
}

fn dema_settle(p: &[ParamVal], eps: f64) -> usize {
    int_param(p, 0).map_or(0, |v| MaKind::Dema.settle(v, eps))
}
fn tema_settle(p: &[ParamVal], eps: f64) -> usize {
    int_param(p, 0).map_or(0, |v| MaKind::Tema.settle(v, eps))
}
fn kama_settle(p: &[ParamVal], eps: f64) -> usize {
    int_param(p, 0).map_or(0, |v| MaKind::Kama.settle(v, eps))
}
fn t3_settle(p: &[ParamVal], eps: f64) -> usize {
    int_param(p, 0).map_or(0, |v| MaKind::T3.settle(v, eps))
}

fn ma_settle(p: &[ParamVal], eps: f64) -> usize {
    match (int_param(p, 0), kind_param(p, 1)) {
        (Ok(period), Ok(kind)) => kind.settle(period, eps),
        _ => 0,
    }
}

fn bbands_settle(p: &[ParamVal], eps: f64) -> usize {
    match (int_param(p, 0), kind_param(p, 3)) {
        (Ok(period), Ok(kind)) => kind.settle(period, eps),
        _ => 0,
    }
}

fn mavp_settle(p: &[ParamVal], eps: f64) -> usize {
    match (int_param(p, 1), kind_param(p, 2)) {
        (Ok(max), Ok(kind)) => kind.settle(max, eps),
        _ => 0,
    }
}

fn wilder_settle(p: &[ParamVal], eps: f64) -> usize {
    // Seed influence decays as (1 - 1/period)^k.
    let period = int_param(p, 0).unwrap_or(1) as f64;
    (eps.ln() / (1.0 - 1.0 / period).ln()).ceil().max(0.0) as usize
}

/// A reversal can happen at any bar, and where the next one lands depends on the whole path: the
/// state never forgets, so there is no settle point short of the period limit.
fn sar_settle(_: &[ParamVal], _: f64) -> usize {
    MAX_PERIOD as usize
}

fn make_ma_fn(
    kind: MaKind,
    vfactor: f64,
    p: usize,
    ins: &[usize],
    outs: &[usize],
) -> Result<Box<dyn Node>, String> {
    let k = MaFn { core: MaCore::with_vfactor(kind, p, vfactor)? };
    Ok(Box::new(Adapter::new(k, regs::<1>(ins, "inputs")?, regs::<1>(outs, "outputs")?)))
}

fn make_wma(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    make_ma_fn(MaKind::Wma, 0.7, int_param(p, 0)?, ins, outs)
}
fn make_dema(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    make_ma_fn(MaKind::Dema, 0.7, int_param(p, 0)?, ins, outs)
}
fn make_tema(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    make_ma_fn(MaKind::Tema, 0.7, int_param(p, 0)?, ins, outs)
}
fn make_trima(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    make_ma_fn(MaKind::Trima, 0.7, int_param(p, 0)?, ins, outs)
}
fn make_kama(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    make_ma_fn(MaKind::Kama, 0.7, int_param(p, 0)?, ins, outs)
}
fn make_t3(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    make_ma_fn(MaKind::T3, float_param(p, 1)?, int_param(p, 0)?, ins, outs)
}
fn make_ma(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    make_ma_fn(kind_param(p, 1)?, 0.7, int_param(p, 0)?, ins, outs)
}

fn make_wilder(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let period = int_param(p, 0)?;
    let k = Wilder { st: EmaSt::with_alpha(period, 1.0 / period.max(1) as f64) };
    Ok(Box::new(Adapter::new(
        k,
        regs::<1>(ins, "wilder inputs")?,
        regs::<1>(outs, "wilder outputs")?,
    )))
}

fn make_bbands(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let period = int_param(p, 0)?;
    let kind = kind_param(p, 3)?;
    let ma = if kind == MaKind::Sma { None } else { Some(MaCore::new(kind, period)?) };
    if kind == MaKind::Sma && period == 0 {
        return Err("bbands period must be at least 1".into());
    }
    let k = BBands {
        window: RingSma::new(period.max(1)),
        var: ShiftedWin::new(period.max(1)),
        ma,
        up: float_param(p, 1)?,
        dn: float_param(p, 2)?,
    };
    Ok(Box::new(Adapter::new(
        k,
        regs::<1>(ins, "bbands inputs")?,
        regs::<3>(outs, "bbands outputs")?,
    )))
}

fn make_mavp(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let (min, max) = (int_param(p, 0)?, int_param(p, 1)?);
    if min == 0 || min > max {
        return Err(format!("mavp needs 1 <= minperiod <= maxperiod, got {min} and {max}"));
    }
    let kind = kind_param(p, 2)?;
    let cores = (min..=max).map(|q| MaCore::new(kind, q)).collect::<Result<Vec<_>, _>>()?;
    let first = kind.lookback(max);
    let starts = (min..=max).map(|q| first - kind.lookback(q)).collect();
    let k = Mavp { min, max, cores, starts, n: 0 };
    Ok(Box::new(Adapter::new(k, regs::<2>(ins, "mavp inputs")?, regs::<1>(outs, "mavp outputs")?)))
}

fn make_midpoint(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = Midpoint(RollMid::new(int_param(p, 0)?.max(1)));
    Ok(Box::new(Adapter::new(
        k,
        regs::<1>(ins, "midpoint inputs")?,
        regs::<1>(outs, "midpoint outputs")?,
    )))
}

fn make_midprice(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let k = Midprice(RollMid::new(int_param(p, 0)?.max(1)));
    Ok(Box::new(Adapter::new(
        k,
        regs::<2>(ins, "midprice inputs")?,
        regs::<1>(outs, "midprice outputs")?,
    )))
}

fn make_sar(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let (acc, max) = (float_param(p, 0)?, float_param(p, 1)?);
    let k = Sar::new(0.0, 0.0, [acc; 2], [acc; 2], [max; 2], false);
    Ok(Box::new(Adapter::new(k, regs::<2>(ins, "sar inputs")?, regs::<1>(outs, "sar outputs")?)))
}

fn make_sarext(p: &[ParamVal], ins: &[usize], outs: &[usize]) -> Result<Box<dyn Node>, String> {
    let f = |i| float_param(p, i);
    let k = Sar::new(f(0)?, f(1)?, [f(2)?, f(5)?], [f(3)?, f(6)?], [f(4)?, f(7)?], true);
    Ok(Box::new(Adapter::new(
        k,
        regs::<2>(ins, "sarext inputs")?,
        regs::<1>(outs, "sarext outputs")?,
    )))
}

pub const DEFS: &[FnDef] = &[
    FnDef {
        name: "sma",
        family: Family::Overlap,
        inputs: INPUT_X,
        params: PERIOD,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_period,
        memory: Memory::Finite,
        legacy: Some(Legacy {
            nan: NanPolicy::Recover,
            pinned_to: "indicators::streaming::StreamingSma",
            tolerance: 0.0,
        }),
        talib_name: Some("SMA"),
        doc: "Simple moving average of the last `period` values. Bit-for-bit the running sum of `ctx.indicators.sma` (add the new value, subtract the one leaving, divide), so a single very large value (a 1e12 print among prices near 100) leaves its rounding in the sum after it has gone: about one part in 1e16 of the spike, which is a visible error on the average until the series resets. Use `sum(x, period) / period` (compensated and re-summed) when that matters.",
        refs: &[],
        rev: 1,
        invariance: &[Invariance::ScaleEquivariant, Invariance::ShiftEquivariant],
        causal: true,
        nan_rule: NanRule::Windowed,
        domain: Domain::Any,
        reference: Reference::Fixture,
        // Measured against TA-Lib 0.6.4 (legacy_vs_talib.txt): same first valid bar, largest
        // difference 1.4e-13 on prices near 100, i.e. ~1e-15 relative.
        relation: Relation::Exact,
        make: make_sma,
    },
    FnDef {
        name: "ema",
        family: Family::Overlap,
        inputs: INPUT_X,
        params: PERIOD,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_period,
        memory: Memory::Infinite { settle: ema_settle },
        legacy: Some(Legacy {
            nan: NanPolicy::Poison,
            pinned_to: "indicators::streaming::StreamingEma",
            tolerance: 0.0,
        }),
        talib_name: Some("EMA"),
        doc: "Exponential moving average, alpha = 2/(period+1), seeded with the mean of the first `period` values.",
        refs: &[],
        rev: 1,
        invariance: &[Invariance::ScaleEquivariant, Invariance::ShiftEquivariant],
        causal: true,
        nan_rule: NanRule::Recursive,
        domain: Domain::Any,
        reference: Reference::Fixture,
        // SMA-seeded like TA-Lib's classic EMA: same first valid bar and, measured, largest
        // difference 5.3e-13 on prices near 100 at periods 14 to 200, so no seed gap to decay.
        relation: Relation::Exact,
        make: make_ema,
    },
    FnDef {
        name: "wma",
        family: Family::Overlap,
        inputs: INPUT_X,
        params: PERIOD,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_period,
        memory: Memory::Finite,
        legacy: Some(Legacy { nan: NanPolicy::Recover, pinned_to: "indicators::streaming::StreamingWma", tolerance: 1e-12 }),
        talib_name: Some("WMA"),
        doc: "Linear-weighted moving average of the last `period` values, newest weighted heaviest. Kept as running sums of deviations from a reference near the window's mean, rebuilt exactly when the reference drifts, a large value leaves or every max(period, 1024) bars; equal to the streaming core to 1e-12 of the window's scale, not bit for bit.",
        refs: &[],
        rev: 2,
        invariance: &[Invariance::ScaleEquivariant, Invariance::ShiftEquivariant],
        causal: true,
        nan_rule: NanRule::Windowed,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_wma,
    },
    FnDef {
        name: "dema",
        family: Family::Overlap,
        inputs: INPUT_X,
        params: PERIOD,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: dema_warmup,
        memory: Memory::Infinite { settle: dema_settle },
        legacy: None,
        talib_name: Some("DEMA"),
        doc: "Double exponential moving average: 2 * EMA - EMA of the EMA, each stage seeded with the mean of the previous stage's first `period` values.",
        refs: &["Mulloy, Smoothing Data with Less Lag, 1994"],
        rev: 1,
        invariance: &[Invariance::ScaleEquivariant, Invariance::ShiftEquivariant],
        causal: true,
        nan_rule: NanRule::Recursive,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_dema,
    },
    FnDef {
        name: "tema",
        family: Family::Overlap,
        inputs: INPUT_X,
        params: PERIOD,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: tema_warmup,
        memory: Memory::Infinite { settle: tema_settle },
        legacy: None,
        talib_name: Some("TEMA"),
        doc: "Triple exponential moving average: 3 * EMA1 - 3 * EMA2 + EMA3, each stage seeded with the mean of the previous stage's first `period` values.",
        refs: &["Mulloy, Smoothing Data with Less Lag, 1994"],
        rev: 1,
        invariance: &[Invariance::ScaleEquivariant, Invariance::ShiftEquivariant],
        causal: true,
        nan_rule: NanRule::Recursive,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_tema,
    },
    FnDef {
        name: "trima",
        family: Family::Overlap,
        inputs: INPUT_X,
        params: PERIOD,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: trima_warmup,
        memory: Memory::Finite,
        legacy: None,
        talib_name: Some("TRIMA"),
        doc: "Triangular moving average: a simple average of a simple average, weights rising then falling over `period` values.",
        refs: &[],
        rev: 1,
        invariance: &[Invariance::ScaleEquivariant, Invariance::ShiftEquivariant],
        causal: true,
        nan_rule: NanRule::Windowed,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_trima,
    },
    FnDef {
        name: "kama",
        family: Family::Overlap,
        inputs: INPUT_X,
        params: PERIOD_MIN2,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: kama_warmup,
        memory: Memory::Infinite { settle: kama_settle },
        legacy: None,
        talib_name: Some("KAMA"),
        doc: "Kaufman adaptive moving average: smoothing between a 2-bar and a 30-bar EMA by the efficiency ratio over `period` bars; first value one bar after the first full window. A window whose path length is under 1e-14 counts as flat (ratio 1), as in TA-Lib.",
        refs: &["Kaufman, Smarter Trading, 1995"],
        rev: 2,
        invariance: &[Invariance::ScaleEquivariant, Invariance::ShiftEquivariant],
        causal: true,
        nan_rule: NanRule::Recursive,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_kama,
    },
    FnDef {
        name: "t3",
        family: Family::Overlap,
        inputs: INPUT_X,
        params: T3_PARAMS,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: t3_warmup,
        memory: Memory::Infinite { settle: t3_settle },
        legacy: None,
        talib_name: Some("T3"),
        doc: "Tillson T3: six cascaded EMAs combined by `vfactor`, each stage seeded with the mean of the previous stage's first `period` values.",
        refs: &["Tillson, Smoothing Techniques for More Accurate Signals, 1998"],
        rev: 1,
        invariance: &[Invariance::ScaleEquivariant, Invariance::ShiftEquivariant],
        causal: true,
        nan_rule: NanRule::Recursive,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_t3,
    },
    FnDef {
        name: "ma",
        family: Family::Overlap,
        inputs: INPUT_X,
        params: PERIOD_MATYPE,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: ma_warmup,
        memory: Memory::Infinite { settle: ma_settle },
        legacy: None,
        talib_name: Some("MA"),
        doc: "Moving average chosen by `matype`: 0 SMA, 1 EMA, 2 WMA, 3 DEMA, 4 TEMA, 5 TRIMA, 6 KAMA, 7 MAMA (limits 0.5 and 0.05, whatever the period; see `mama`), 8 T3. Period 1 returns the input. Kinds other than 0, 2 and 5 hold their state on a NaN.",
        refs: &[],
        rev: 3,
        invariance: &[Invariance::ScaleEquivariant, Invariance::ShiftEquivariant],
        causal: true,
        nan_rule: NanRule::Windowed,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_ma,
    },
    FnDef {
        name: "mavp",
        family: Family::Overlap,
        inputs: INPUT_X_PERIODS,
        params: MAVP_PARAMS,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: mavp_warmup,
        memory: Memory::Infinite { settle: mavp_settle },
        legacy: None,
        talib_name: Some("MAVP"),
        doc: "Moving average whose period changes every bar: `periods` (cut to a whole number in minperiod..maxperiod) picks, per bar, which period's `matype` average to read. Each period's average starts so that it takes its first value at the first output bar, as TA-Lib's does, so a recursive kind (matype 1, 3, 4, 6, 8) carries no state from before it; a NaN period gives NaN. Warm-up is the longest period's.",
        refs: &[],
        rev: 3,
        invariance: &[Invariance::ScaleEquivariant, Invariance::ShiftEquivariant],
        causal: true,
        nan_rule: NanRule::Windowed,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_mavp,
    },
    FnDef {
        name: "bbands",
        family: Family::Overlap,
        inputs: INPUT_X,
        params: BBANDS_PARAMS,
        outputs: &["upperband", "middleband", "lowerband"],
        out_kind: OutKind::Series,
        warmup: bbands_warmup,
        memory: Memory::Infinite { settle: bbands_settle },
        legacy: Some(Legacy { nan: NanPolicy::Recover, pinned_to: "indicators::streaming::StreamingBollinger", tolerance: 1e-12 }),
        talib_name: Some("BBANDS"),
        doc: "Bollinger bands: a `matype` average of the last `period` values (default SMA) plus nbdevup and minus nbdevdn population standard deviations of the window around its simple mean. Outputs upper, middle, lower.",
        refs: &["Bollinger, Bollinger on Bollinger Bands, 2001"],
        rev: 3,
        invariance: &[Invariance::ScaleEquivariant, Invariance::ShiftEquivariant],
        causal: true,
        nan_rule: NanRule::Windowed,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_bbands,
    },
    FnDef {
        name: "midpoint",
        family: Family::Overlap,
        inputs: INPUT_X,
        params: PERIOD,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_period,
        memory: Memory::Finite,
        legacy: None,
        talib_name: Some("MIDPOINT"),
        doc: "Midpoint of the range: (highest + lowest) / 2 of the last `period` values.",
        refs: &[],
        rev: 1,
        invariance: &[Invariance::ScaleEquivariant, Invariance::ShiftEquivariant],
        causal: true,
        nan_rule: NanRule::Windowed,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_midpoint,
    },
    FnDef {
        name: "midprice",
        family: Family::Overlap,
        inputs: INPUT_HL,
        params: PERIOD,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_period,
        memory: Memory::Finite,
        legacy: None,
        talib_name: Some("MIDPRICE"),
        doc: "Midpoint of the price range: (highest high + lowest low) / 2 of the last `period` bars.",
        refs: &[],
        rev: 1,
        invariance: &[Invariance::ScaleEquivariant, Invariance::ShiftEquivariant],
        causal: true,
        nan_rule: NanRule::Windowed,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_midprice,
    },
    FnDef {
        name: "sar",
        family: Family::Overlap,
        inputs: INPUT_HL,
        params: SAR_PARAMS,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: sar_warmup,
        memory: Memory::Infinite { settle: sar_settle },
        legacy: None,
        talib_name: Some("SAR"),
        doc: "Parabolic SAR. The first direction comes from the minus directional move of the first two bars; a bar with a NaN high or low is skipped.",
        refs: &["Wilder, New Concepts in Technical Trading Systems, 1978"],
        rev: 1,
        invariance: &[Invariance::ScaleEquivariant, Invariance::ShiftEquivariant],
        causal: true,
        nan_rule: NanRule::Recursive,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_sar,
    },
    FnDef {
        name: "sarext",
        family: Family::Overlap,
        inputs: INPUT_HL,
        params: SAREXT_PARAMS,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: sar_warmup,
        memory: Memory::Infinite { settle: sar_settle },
        legacy: None,
        talib_name: Some("SAREXT"),
        doc: "Parabolic SAR, extended: separate long and short factors, a start value (positive long, negative short, 0 to decide from the first two bars), an offset on reversal; short values are reported negative.",
        refs: &["Wilder, New Concepts in Technical Trading Systems, 1978"],
        rev: 1,
        invariance: &[Invariance::ScaleEquivariant],
        causal: true,
        nan_rule: NanRule::Recursive,
        domain: Domain::Any,
        reference: Reference::Fixture,
        relation: Relation::Exact,
        make: make_sarext,
    },
    FnDef {
        name: "wilder",
        family: Family::Overlap,
        inputs: INPUT_X,
        params: PERIOD,
        outputs: &["value"],
        out_kind: OutKind::Series,
        warmup: warmup_period,
        memory: Memory::Infinite { settle: wilder_settle },
        legacy: Some(Legacy { nan: NanPolicy::Poison, pinned_to: "indicators::streaming::StreamingEma::wilder", tolerance: 0.0 }),
        talib_name: None,
        doc: "Wilder smoothing: alpha = 1/period, seeded with the mean of the first `period` values.",
        refs: &["Wilder, New Concepts in Technical Trading Systems, 1978"],
        rev: 1,
        invariance: &[Invariance::ScaleEquivariant, Invariance::ShiftEquivariant],
        causal: true,
        nan_rule: NanRule::Recursive,
        domain: Domain::Any,
        reference: Reference::Hand("overlap::tests::wilder_hand_value"),
        relation: Relation::NoLibrary,
        make: make_wilder,
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indicators::streaming::{StreamingEma, StreamingIndicator, StreamingSma};
    use crate::quant::kernel::{canon_bits, Regs};
    use crate::quant::registry::lookup;
    use proptest::prelude::*;
    use proptest::test_runner::{Config, RngAlgorithm, RngSeed, TestRunner};

    fn runner(seed: u64) -> TestRunner {
        TestRunner::new(Config {
            cases: 300,
            failure_persistence: None,
            rng_algorithm: RngAlgorithm::ChaCha,
            rng_seed: RngSeed::Fixed(seed),
            ..Config::default()
        })
    }

    fn node(name: &str, period: i64) -> Box<dyn Node> {
        (lookup(name).unwrap().make)(&[ParamVal::Int(period)], &[0], &[1]).unwrap()
    }

    fn batch(name: &str, period: i64, data: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; data.len()];
        node(name, period).run_batch(&[data], &mut [&mut out[..]]);
        out
    }

    fn stream(name: &str, period: i64, data: &[f64]) -> Vec<f64> {
        let mut n = node(name, period);
        let mut regs = Regs::new(2);
        data.iter()
            .map(|&x| {
                regs.set(0, x);
                n.step(&mut regs);
                regs.get(1)
            })
            .collect()
    }

    fn bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|&x| canon_bits(x)).collect()
    }

    #[test]
    fn batch_and_streaming_are_bit_identical_with_nans() {
        let value = prop_oneof![8 => -1e6f64..1e6, 1 => Just(f64::NAN), 1 => Just(f64::INFINITY)];
        let strategy = (1i64..40, prop::collection::vec(value, 0..300));
        runner(11)
            .run(&strategy, |(p, data)| {
                for name in ["sma", "ema"] {
                    prop_assert_eq!(
                        bits(&batch(name, p, &data)),
                        bits(&stream(name, p, &data)),
                        "{} p={}",
                        name,
                        p
                    );
                }
                Ok(())
            })
            .unwrap();
    }

    fn legacy(core: &mut dyn StreamingIndicator, data: &[f64]) -> Vec<f64> {
        data.iter().map(|&x| core.update(x).unwrap_or(f64::NAN)).collect()
    }

    #[test]
    fn sma_and_ema_equal_the_streaming_cores_bitwise_on_finite_data() {
        let strategy = (1i64..40, prop::collection::vec(-1e6f64..1e6, 0..300));
        runner(12)
            .run(&strategy, |(p, data)| {
                let sma = legacy(&mut StreamingSma::new(p as usize), &data);
                let ema = legacy(&mut StreamingEma::new(p as usize), &data);
                prop_assert_eq!(bits(&batch("sma", p, &data)), bits(&sma));
                prop_assert_eq!(bits(&stream("sma", p, &data)), bits(&sma));
                prop_assert_eq!(bits(&batch("ema", p, &data)), bits(&ema));
                prop_assert_eq!(bits(&stream("ema", p, &data)), bits(&ema));
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn warmup_formula_matches_first_non_nan() {
        let data: Vec<f64> = (1..=30).map(f64::from).collect();
        for name in ["sma", "ema"] {
            for p in [1i64, 2, 5, 14] {
                let first = batch(name, p, &data).iter().position(|v| !v.is_nan()).unwrap();
                let def = lookup(name).unwrap();
                assert_eq!(first, (def.warmup)(&[ParamVal::Int(p)]), "{name} p={p}");
            }
        }
    }

    #[test]
    fn sma_hand_value() {
        // (1 + 2 + 3) / 3 = 2 and (2 + 3 + 4) / 3 = 3.
        let out = batch("sma", 3, &[1.0, 2.0, 3.0, 4.0]);
        assert!(out[0].is_nan() && out[1].is_nan());
        assert_eq!((out[2], out[3]), (2.0, 3.0));
    }

    #[test]
    fn windowed_recovers_after_the_nan_leaves() {
        let out = batch("sma", 3, &[1.0, 2.0, 3.0, f64::NAN, 5.0, 6.0, 7.0, 8.0]);
        // The window holds the NaN at indices 3, 4, 5; index 6 is (5 + 6 + 7) / 3.
        assert!(out[3].is_nan() && out[4].is_nan() && out[5].is_nan());
        assert_eq!(out[6], 6.0);
        assert_eq!(out[7], 7.0);
    }

    #[test]
    fn recursive_holds_state_on_nan_without_poisoning() {
        // Leading NaNs (an upstream warm-up) do not disturb the seed.
        let nan = f64::NAN;
        let clean = batch("ema", 3, &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let lead = batch("ema", 3, &[nan, nan, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(bits(&lead[2..]), bits(&clean));
        assert!(lead[0].is_nan() && lead[1].is_nan());
        // A NaN mid-stream outputs NaN, and the next value continues from the held state:
        // seed (1 + 2 + 3) / 3 = 2, alpha = 0.5, then 0.5 * 4 + 0.5 * 2 = 3.
        let gap = batch("ema", 3, &[1.0, 2.0, 3.0, nan, 4.0]);
        assert_eq!(gap[2], 2.0);
        assert!(gap[3].is_nan());
        assert_eq!(gap[4], 3.0);
    }
}

#[cfg(test)]
mod family_tests {
    use super::*;
    use crate::indicators::streaming::{
        StreamingBollinger, StreamingEma, StreamingIndicator, StreamingWma,
    };
    use crate::quant::kernel::{canon_bits, Regs};
    use crate::quant::registry::lookup;
    use proptest::prelude::*;
    use proptest::test_runner::{Config, RngAlgorithm, RngSeed, TestRunner};

    const NAN: f64 = f64::NAN;

    fn runner(seed: u64) -> TestRunner {
        TestRunner::new(Config {
            cases: 200,
            failure_persistence: None,
            rng_algorithm: RngAlgorithm::ChaCha,
            rng_seed: RngSeed::Fixed(seed),
            ..Config::default()
        })
    }

    fn node(name: &str, params: &[ParamVal], n_in: usize, n_out: usize) -> Box<dyn Node> {
        let ins: Vec<usize> = (0..n_in).collect();
        let outs: Vec<usize> = (n_in..n_in + n_out).collect();
        (lookup(name).unwrap().make)(params, &ins, &outs).unwrap()
    }

    /// Batch: one `Vec` per output.
    fn batch(name: &str, params: &[ParamVal], cols: &[&[f64]], n_out: usize) -> Vec<Vec<f64>> {
        let n = cols[0].len();
        let mut outs = vec![vec![0.0; n]; n_out];
        let mut refs: Vec<&mut [f64]> = outs.iter_mut().map(|o| &mut o[..]).collect();
        node(name, params, cols.len(), n_out).run_batch(cols, &mut refs);
        outs
    }

    fn stream(name: &str, params: &[ParamVal], cols: &[&[f64]], n_out: usize) -> Vec<Vec<f64>> {
        let n_in = cols.len();
        let mut nd = node(name, params, n_in, n_out);
        let mut regs = Regs::new(n_in + n_out);
        let mut outs = vec![Vec::new(); n_out];
        for i in 0..cols[0].len() {
            for (j, c) in cols.iter().enumerate() {
                regs.set(j, c[i]);
            }
            nd.step(&mut regs);
            for (o, v) in outs.iter_mut().enumerate() {
                v.push(regs.get(n_in + o));
            }
        }
        outs
    }

    fn one(name: &str, params: &[ParamVal], x: &[f64]) -> Vec<f64> {
        batch(name, params, &[x], 1).remove(0)
    }

    fn bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|&x| canon_bits(x)).collect()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * b.abs().max(1.0)
    }

    fn i(v: i64) -> ParamVal {
        ParamVal::Int(v)
    }

    fn f(v: f64) -> ParamVal {
        ParamVal::Float(v)
    }

    /// Every new function with parameters and input arity, for the mode-agreement and warm-up
    /// checks.
    fn cases() -> Vec<(&'static str, Vec<ParamVal>, usize, usize)> {
        let mut v = vec![
            ("wma", vec![i(5)], 1, 1),
            ("dema", vec![i(4)], 1, 1),
            ("tema", vec![i(4)], 1, 1),
            ("trima", vec![i(5)], 1, 1),
            ("trima", vec![i(6)], 1, 1),
            ("kama", vec![i(5)], 1, 1),
            ("t3", vec![i(3), f(0.7)], 1, 1),
            ("bbands", vec![i(5), f(2.0), f(1.5), i(0)], 1, 3),
            ("bbands", vec![i(5), f(2.0), f(2.0), i(1)], 1, 3),
            ("midpoint", vec![i(4)], 1, 1),
            ("midprice", vec![i(4)], 2, 1),
            ("sar", vec![f(0.02), f(0.2)], 2, 1),
            (
                "sarext",
                vec![f(0.0), f(0.01), f(0.03), f(0.04), f(0.3), f(0.02), f(0.03), f(0.25)],
                2,
                1,
            ),
            ("wilder", vec![i(4)], 1, 1),
            ("mavp", vec![i(2), i(5), i(0)], 2, 1),
            ("mavp", vec![i(2), i(5), i(1)], 2, 1),
        ];
        for m in [0, 1, 2, 3, 4, 5, 6, 7, 8] {
            v.push(("ma", vec![i(5), i(m)], 1, 1));
        }
        v
    }

    #[test]
    fn batch_and_streaming_are_bit_identical_with_nans() {
        let value = prop_oneof![8 => 1.0f64..200.0, 1 => Just(NAN), 1 => Just(f64::INFINITY)];
        let strategy = prop::collection::vec((value.clone(), value.clone(), value), 0..160);
        runner(21)
            .run(&strategy, |rows| {
                let a: Vec<f64> = rows.iter().map(|r| r.0).collect();
                let b: Vec<f64> = rows.iter().map(|r| r.1).collect();
                let per: Vec<f64> = rows.iter().map(|r| r.2 % 7.0).collect();
                for (name, params, n_in, n_out) in cases() {
                    let cols: Vec<&[f64]> = match (name, n_in) {
                        ("mavp", _) => vec![&a, &per],
                        (_, 1) => vec![&a],
                        _ => vec![&a, &b],
                    };
                    let (x, y) =
                        (batch(name, &params, &cols, n_out), stream(name, &params, &cols, n_out));
                    for o in 0..n_out {
                        prop_assert_eq!(
                            bits(&x[o]),
                            bits(&y[o]),
                            "{} {:?} output {}",
                            name,
                            params,
                            o
                        );
                    }
                }
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn warmup_formula_matches_first_non_nan() {
        let x: Vec<f64> = (1..=80).map(|v| f64::from(v) + f64::from(v % 5)).collect();
        let hi: Vec<f64> = x.iter().map(|v| v + 1.0).collect();
        let lo: Vec<f64> = x.iter().map(|v| v - 1.0).collect();
        let per = vec![4.0; x.len()];
        for (name, params, n_in, n_out) in cases() {
            let cols: Vec<&[f64]> = match (name, n_in) {
                ("mavp", _) => vec![&x, &per],
                (_, 1) => vec![&x],
                _ => vec![&hi, &lo],
            };
            let out = &batch(name, &params, &cols, n_out)[0];
            let first = out.iter().position(|v| !v.is_nan()).unwrap();
            assert_eq!(first, (lookup(name).unwrap().warmup)(&params), "{name} {params:?}");
        }
    }

    #[test]
    fn wma_hand_value_and_legacy_within_tolerance() {
        // (1 * 1 + 2 * 2 + 3 * 3) / 6 = 14 / 6.
        let out = one("wma", &[i(3)], &[1.0, 2.0, 3.0]);
        assert!(out[0].is_nan() && out[1].is_nan());
        assert!(close(out[2], 14.0 / 6.0));
        let strategy = (1i64..30, prop::collection::vec(-1e6f64..1e6, 0..200));
        runner(22)
            .run(&strategy, |(p, data)| {
                let mut core = StreamingWma::new(p as usize);
                let legacy: Vec<f64> =
                    data.iter().map(|&x| core.update(x).unwrap_or(NAN)).collect();
                // Pinned at 1e-12 of the window's scale (`Legacy.tolerance`), not bitwise: this
                // is a running sum, the legacy core a fresh weighted sum every bar.
                let got = one("wma", &[i(p)], &data);
                let p = p as usize;
                for (j, (g, l)) in got.iter().zip(&legacy).enumerate() {
                    if l.is_nan() || j + 1 < p {
                        prop_assert!(g.is_nan() && l.is_nan(), "bar {j}");
                        continue;
                    }
                    let scale = data[j + 1 - p..=j].iter().fold(0.0f64, |m, v| m.max(v.abs()));
                    prop_assert!((g - l).abs() <= 1e-12 * scale, "bar {j}: {g} against {l}");
                }
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn wilder_hand_value() {
        // period 2, alpha 0.5: seed (1 + 2) / 2 = 1.5, then 0.5 * 3 + 0.5 * 1.5 = 2.25, then
        // 0.5 * 4 + 0.5 * 2.25 = 3.125 (all exact in binary).
        let out = one("wilder", &[i(2)], &[1.0, 2.0, 3.0, 4.0]);
        assert!(out[0].is_nan());
        assert_eq!(out[1..], [1.5, 2.25, 3.125]);
    }

    #[test]
    fn wilder_equals_the_streaming_core_bitwise() {
        let strategy = (1i64..30, prop::collection::vec(-1e6f64..1e6, 0..200));
        runner(23)
            .run(&strategy, |(p, data)| {
                let mut core = StreamingEma::wilder(p as usize);
                let legacy: Vec<f64> =
                    data.iter().map(|&x| core.update(x).unwrap_or(NAN)).collect();
                prop_assert_eq!(bits(&one("wilder", &[i(p)], &data)), bits(&legacy));
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn dema_and_tema_hand_values() {
        // period 2, alpha 2/3, on 1..5. EMA1: seed 1.5 at bar 1, then 2.5, 3.5, 4.5.
        // EMA2 seeds with (1.5 + 2.5) / 2 = 2 at bar 2, then 2/3 * 3.5 + 1/3 * 2 = 3, then 4.
        // DEMA = 2 * EMA1 - EMA2 = 3, 4, 5 at bars 2, 3, 4.
        let d = one("dema", &[i(2)], &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(d[..2].iter().all(|v| v.is_nan()));
        assert!(close(d[2], 3.0) && close(d[3], 4.0) && close(d[4], 5.0));
        // EMA3 seeds with (2 + 3) / 2 = 2.5 at bar 3; TEMA = EMA3 + (3 * 3.5 - 3 * 3) = 4.
        let t = one("tema", &[i(2)], &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(t[..3].iter().all(|v| v.is_nan()));
        assert!(close(t[3], 4.0));
    }

    #[test]
    fn dema_holds_state_across_a_nan_and_leading_nans() {
        let clean = one("dema", &[i(2)], &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let lead = one("dema", &[i(2)], &[NAN, NAN, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(bits(&lead[2..]), bits(&clean));
        let gap = one("dema", &[i(2)], &[1.0, 2.0, 3.0, NAN, 4.0]);
        assert!(gap[3].is_nan());
        assert_eq!(bits(&gap[4..]), bits(&clean[3..4]));
    }

    #[test]
    fn t3_of_a_constant_is_the_constant() {
        // The four coefficients sum to 1 for any vfactor: -a3 + (3a2 + 3a3) + (-6a2 - 3a - 3a3)
        // + (1 + 3a + a3 + 3a2) = 1.
        for a in [0.0, 0.3, 0.7, 1.0] {
            let out = one("t3", &[i(3), f(a)], &[5.0; 20]);
            assert!(out[..12].iter().all(|v| v.is_nan()));
            assert!(out[12..].iter().all(|v| close(*v, 5.0)), "a={a} {out:?}");
        }
    }

    #[test]
    fn trima_hand_values() {
        // period 3 is SMA(2) of SMA(2): weights 1 2 1 over 4, so (1 + 2 * 2 + 3) / 4 = 2.
        let o = one("trima", &[i(3)], &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(o[..2].iter().all(|v| v.is_nan()));
        assert!(close(o[2], 2.0) && close(o[3], 3.0) && close(o[4], 4.0));
        // period 4 is SMA(2) of SMA(3), weights 1 2 2 1 over 6: (1 + 4 + 6 + 4) / 6 = 2.5.
        let e = one("trima", &[i(4)], &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(e[..3].iter().all(|v| v.is_nan()));
        assert!(close(e[3], 2.5));
    }

    #[test]
    fn trima_windowed_recovers_after_a_nan() {
        let o = one("trima", &[i(3)], &[1.0, 2.0, 3.0, NAN, 5.0, 6.0, 7.0, 8.0, 9.0]);
        // Index 3 is NaN; the two-stage window spans 3 bars, so 3, 4, 5 are NaN and 6 is
        // (5 + 2 * 6 + 7) / 4 = 6.
        assert!(o[3..6].iter().all(|v| v.is_nan()));
        assert!(close(o[6], 6.0));
    }

    #[test]
    fn kama_hand_value() {
        // period 2 on 1, 2, 4: sum of |differences| = 1 + 2 = 3, change over 2 bars = 3, so the
        // efficiency ratio is 1 and the constant is (2/3)^2 = 4/9. Seed is the bar before:
        // 2 + (4 - 2) * 4/9 = 26/9.
        let o = one("kama", &[i(2)], &[1.0, 2.0, 4.0]);
        assert!(o[0].is_nan() && o[1].is_nan());
        assert!(close(o[2], 26.0 / 9.0));
    }

    #[test]
    fn kama_treats_a_window_flatter_than_1e_14_as_flat() {
        // period 2 on 0, 1e-16, 0: the path is 2e-16 and the change 0, so the ratio would be 0,
        // but the window is flat and the ratio is 1. Constant (2/3)^2 = 4/9 on the seed 1e-16:
        // 1e-16 + (0 - 1e-16) * 4/9 = 5/9 * 1e-16. A ratio of 0 would give 2/31 squared
        // and about 9.96e-17.
        let o = one("kama", &[i(2)], &[0.0, 1e-16, 0.0]);
        assert!((o[2] - 5.0 / 9.0 * 1e-16).abs() < 1e-30, "{}", o[2]);
    }

    #[test]
    fn midpoint_and_midprice_hand_values() {
        let o = one("midpoint", &[i(3)], &[1.0, 5.0, 3.0, 2.0]);
        assert!(o[0].is_nan() && o[1].is_nan());
        // (5 + 1) / 2, then (5 + 2) / 2.
        assert_eq!((o[2], o[3]), (3.0, 3.5));
        let hi = [3.0, 4.0, 5.0, 6.0];
        let lo = [1.0, 2.0, 1.0, 2.0];
        let m = &batch("midprice", &[i(2)], &[&hi, &lo], 1)[0];
        // (4 + 1) / 2, (5 + 1) / 2, (6 + 1) / 2.
        assert!(m[0].is_nan());
        assert_eq!(m[1..], [2.5, 3.0, 3.5]);
    }

    #[test]
    fn midpoint_recovers_after_a_nan_leaves() {
        let o = one("midpoint", &[i(2)], &[1.0, 3.0, NAN, 5.0, 7.0]);
        assert!(o[2].is_nan() && o[3].is_nan());
        assert_eq!(o[4], 6.0);
    }

    #[test]
    fn bbands_hand_value() {
        // period 3 on 1, 2, 3, 4: mean 2, population variance (1 + 0 + 1) / 3, deviation
        // sqrt(2/3); upper = 2 + 2 * sd, lower = 2 - 1.5 * sd.
        let x = [1.0, 2.0, 3.0, 4.0];
        let o = batch("bbands", &[i(3), f(2.0), f(1.5), i(0)], &[&x], 3);
        let sd = (2.0f64 / 3.0).sqrt();
        assert!(o.iter().all(|c| c[0].is_nan() && c[1].is_nan()));
        assert!(close(o[0][2], 2.0 + 2.0 * sd) && o[1][2] == 2.0 && close(o[2][2], 2.0 - 1.5 * sd));
        assert!(o[1][3] == 3.0);
    }

    #[test]
    fn bbands_equals_the_streaming_bollinger_on_finite_data() {
        let strategy = (1i64..30, 0.5f64..3.0, prop::collection::vec(-1e4f64..1e4, 0..200));
        runner(24)
            .run(&strategy, |(p, k, data)| {
                let mut core = StreamingBollinger::new(p as usize, k);
                let legacy: Vec<(f64, f64, f64)> = {
                    let mut last = (NAN, NAN, NAN);
                    data.iter()
                        .map(|&x| {
                            if let Some(v) = core.update(x) {
                                last = v;
                            }
                            last
                        })
                        .collect()
                };
                let o = batch("bbands", &[i(p), f(k), f(k), i(0)], &[&data], 3);
                let full = (p as usize).saturating_sub(1);
                for (j, (mid, up, lo)) in legacy.iter().enumerate().skip(full) {
                    // The middle band is the legacy running mean, bit for bit; the bands add a
                    // deviation from the O(1) window sums, pinned at 1e-12 (`Legacy.tolerance`).
                    prop_assert_eq!(canon_bits(o[1][j]), canon_bits(*mid), "middle {}", j);
                    prop_assert!(close(o[0][j], *up), "upper {}: {} against {}", j, o[0][j], up);
                    prop_assert!(close(o[2][j], *lo), "lower {}: {} against {}", j, o[2][j], lo);
                }
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn bbands_middle_band_follows_matype() {
        let x: Vec<f64> = (1..=30).map(|v| f64::from(v * v % 17) + 10.0).collect();
        let ema = one("ema", &[i(5)], &x);
        let o = batch("bbands", &[i(5), f(2.0), f(2.0), i(1)], &[&x], 3);
        assert_eq!(bits(&o[1]), bits(&ema));
    }

    #[test]
    fn ma_selects_the_average_by_matype() {
        let x: Vec<f64> = (1..=40).map(|v| f64::from(v * 7 % 13) + 20.0).collect();
        for (code, name) in [
            (0, "sma"),
            (1, "ema"),
            (2, "wma"),
            (3, "dema"),
            (4, "tema"),
            (5, "trima"),
            (6, "kama"),
        ] {
            assert_eq!(
                bits(&one("ma", &[i(6), i(code)], &x)),
                bits(&one(name, &[i(6)], &x)),
                "matype {code} = {name}"
            );
        }
        assert_eq!(bits(&one("ma", &[i(6), i(8)], &x)), bits(&one("t3", &[i(6), f(0.7)], &x)));
        // Period 1 is the input itself.
        assert_eq!(bits(&one("ma", &[i(1), i(1)], &x)), bits(&x));
    }

    #[test]
    fn ma_matype_7_is_mama_at_fixed_limits() {
        // TA-Lib's MA with matype 7 ignores the period: it is MAMA at 0.5 and 0.05.
        let x: Vec<f64> = (1..=80).map(|v| f64::from(v * 7 % 13) + 20.0).collect();
        let want = &batch("mama", &[f(0.5), f(0.05)], &[&x], 2)[0];
        assert_eq!(bits(&one("ma", &[i(5), i(7)], &x)), bits(want));
        assert_eq!(bits(&one("ma", &[i(30), i(7)], &x)), bits(want));
        assert_eq!(bits(&one("ma", &[i(1), i(7)], &x)), bits(&x));
        assert!(MaKind::from_code(9).is_err());
    }

    #[test]
    fn mavp_reads_the_period_of_each_bar() {
        // minperiod 2, maxperiod 3, SMA. Warm-up is the longest period's (index 2).
        // Bar 2 uses period 2: (2 + 3) / 2; bar 3 period 3: (2 + 3 + 4) / 3; bar 4 period 2:
        // (4 + 5) / 2. Periods outside the range are cut to it (9 -> 3, 0.5 -> 2).
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let per = [2.0, 2.0, 2.0, 3.0, 2.0, 9.0, 0.5];
        let o = &batch("mavp", &[i(2), i(3), i(0)], &[&x, &per], 1)[0];
        assert!(o[0].is_nan() && o[1].is_nan());
        assert_eq!(o[2], 2.5);
        assert_eq!(o[3], 3.0);
        assert_eq!(o[4], 4.5);
        assert_eq!(o[5], 5.0);
        assert_eq!(o[6], 6.5);
        // A NaN period gives NaN for that bar only.
        let per = [3.0, 3.0, 3.0, NAN, 3.0];
        let o = &batch("mavp", &[i(2), i(3), i(0)], &[&x[..5], &per], 1)[0];
        assert!(o[3].is_nan() && o[4] == 4.0);
    }

    #[test]
    fn mavp_seeds_a_recursive_kind_at_the_first_output_bar() {
        // minperiod 2, maxperiod 3, EMA: the first output is bar 2 (lookback of period 3). The
        // period-2 average starts at bar 1, seeded with (0 + 3) / 2 = 1.5 at bar 2, then
        // 2/3 * 9 + 1/3 * 1.5 = 6.5 at bar 3. Run from bar 0 it would be seeded at bar 1 with
        // (10 + 0) / 2 = 5 and give 11/3 at bar 2: TA-Lib's MAVP does not.
        let o = &batch("mavp", &[i(2), i(3), i(1)], &[&[10.0, 0.0, 3.0, 9.0], &[2.0; 4]], 1)[0];
        assert!(o[0].is_nan() && o[1].is_nan());
        assert_eq!(o[2], 1.5);
        assert!(close(o[3], 6.5));
    }

    #[test]
    fn sar_hand_values_and_the_first_direction_rule() {
        // Bars (high, low): (10, 8), (11, 9), (12, 10), (13, 11). Up move: the minus move of
        // bars 0, 1 is 0, so long, sar = low of bar 0 = 8, ep = 11, af = 0.02.
        // Bar 1 outputs 8, then sar = 8 + 0.02 * (11 - 8) = 8.06. Bar 2: low 10 > 8.06, outputs
        // 8.06; high 12 > ep so ep = 12, af = 0.04; sar = 8.06 + 0.04 * (12 - 8.06) = 8.2176.
        let hi = [10.0, 11.0, 12.0, 13.0];
        let lo = [8.0, 9.0, 10.0, 11.0];
        let o = &batch("sar", &[f(0.02), f(0.2)], &[&hi, &lo], 1)[0];
        assert!(o[0].is_nan());
        assert!(close(o[1], 8.0) && close(o[2], 8.06) && close(o[3], 8.2176));
        // Down move: bars (10, 8), (9, 7). Minus move = 8 - 7 = 1 > 0 and above the plus move
        // -1, so short, sar = high of bar 0 = 10, ep = 7. Bar 2 (11, 8): high 11 >= 9.94
        // (= 10 + 0.02 * (7 - 10)), so the trend reverses to long and the value is the old
        // extreme 7 (not above bar 1's low 7 or bar 2's low 8). SAREXT reports the short value
        // negative.
        let hi = [10.0, 9.0, 11.0];
        let lo = [8.0, 7.0, 8.0];
        let s = &batch("sar", &[f(0.02), f(0.2)], &[&hi, &lo], 1)[0];
        assert!(s[0].is_nan() && s[1] == 10.0 && s[2] == 7.0);
        let params = [f(0.0), f(0.0), f(0.02), f(0.02), f(0.2), f(0.02), f(0.02), f(0.2)];
        let e = &batch("sarext", &params, &[&hi, &lo], 1)[0];
        assert!(e[0].is_nan() && e[1] == -10.0 && e[2] == 7.0);
    }

    #[test]
    fn sar_skips_a_nan_bar_and_continues() {
        let hi = [10.0, 11.0, NAN, 12.0, 13.0];
        let lo = [8.0, 9.0, 9.5, 10.0, 11.0];
        let o = &batch("sar", &[f(0.02), f(0.2)], &[&hi, &lo], 1)[0];
        assert!(o[2].is_nan());
        // The bar after the gap continues from the state of bar 1, as if the NaN bar were absent.
        let clean = &batch(
            "sar",
            &[f(0.02), f(0.2)],
            &[&[10.0, 11.0, 12.0, 13.0], &[8.0, 9.0, 10.0, 11.0]],
            1,
        )[0];
        assert_eq!((o[3], o[4]), (clean[2], clean[3]));
    }

    #[test]
    fn ma_core_steps_like_the_function() {
        let x: Vec<f64> = (1..=30).map(|v| f64::from(v * 5 % 11) + 3.0).collect();
        let mut core = MaCore::new(MaKind::Tema, 4).unwrap();
        let stepped: Vec<f64> = x.iter().map(|&v| core.step(v)).collect();
        assert_eq!(bits(&stepped), bits(&one("tema", &[i(4)], &x)));
        assert!(core.warmed());
    }
}
